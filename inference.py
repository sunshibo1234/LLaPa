import os
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# --- 全局常量和配置 (Global Constants and Configuration) ---

# 定义模型和图像的常量
# Define constants for model and images
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 448
MAX_IMAGE_PATCHES = 12 # 与InternVL-Chat-V1.5-Vision-Res-448px模型匹配

# --- 图像预处理辅助函数 (Image Preprocessing Helper Functions) ---

def build_transform(input_size):
    """
    构建图像预处理的转换流程。
    Builds the transformation pipeline for image preprocessing.
    """
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    为给定图片找到最接近的目标宽高比。
    Finds the closest target aspect ratio for a given image.
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    动态地将图像分割成多个块以适应模型输入。
    Dynamically splits an image into multiple patches to fit the model's input requirements.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
        
    return processed_images

def load_images_and_masks(image_dir, image_size=IMAGE_SIZE, max_num=MAX_IMAGE_PATCHES):
    """
    从文件夹加载所有图片及其对应的mask。
    Loads all images and their corresponding masks from a directory.
    """
    transform = build_transform(input_size=image_size)
    pixel_values_list = []
    num_patches_list = []
    masks = []
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for img_file in image_files:
        image_path = os.path.join(image_dir, img_file)
        
        # 加载图片
        try:
            image = Image.open(image_path).convert('RGB')
            processed_imgs = dynamic_preprocess(image, image_size=image_size, use_thumbnail=True, max_num=max_num)
            
            pixel_values = [transform(img) for img in processed_imgs]
            pixel_values_list.extend(pixel_values)
            num_patches_list.append(len(pixel_values))
        except IOError:
            print(f"Warning: Cannot open or read image file at {image_path}. Skipping.")
            continue

        # 加载对应的mask
        base_name = os.path.splitext(img_file)[0]
        mask_path = os.path.join(image_dir, f"{base_name}_mask.npy")
        if os.path.exists(mask_path):
            try:
                mask = torch.from_numpy(np.load(mask_path)).float()
                masks.append(mask)
            except Exception as e:
                print(f"Warning: Could not load or process mask {mask_path}. Using a default mask. Error: {e}")
                masks.append(torch.ones((1, image.height, image.width))) # 使用原始图片尺寸
        else:
            print(f"Warning: Mask file not found at {mask_path}. Using a default mask.")
            masks.append(torch.ones((1, image.height, image.width))) # 使用原始图片尺寸

    if not pixel_values_list:
        return None, None, None

    # 将数据转换为张量
    pixel_values_tensor = torch.stack(pixel_values_list)
    
    # 将所有mask调整到相同大小并堆叠
    # 注意：这里我们假设所有原始图片尺寸相同，如果不同，需要更复杂的处理
    # For simplicity, we assume all original images have the same size.
    # If not, more complex processing (like padding/resizing masks) is needed.
    squeezed_masks = [m.squeeze(0) for m in masks]
    image_mask_tensor = torch.stack(squeezed_masks, dim=0)

    return pixel_values_tensor, num_patches_list, image_mask_tensor


# --- 推理主类 (Main Inference Class) ---

class SingleInstanceInfer:
    """
    一个封装了模型加载和推理过程的类。
    A class that encapsulates the model loading and inference process.
    """
    def __init__(self, model_path, torch_dtype=torch.bfloat16):
        """
        初始化模型和分词器。
        Initializes the model and tokenizer.
        """
        print("Loading model and tokenizer...")
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.device = next(self.model.parameters()).device
        print("Model and tokenizer loaded successfully.")

    def predict(self, image_dir, task_file_path):
        """
        执行单次推理。
        Performs a single inference run.
        """
        # 1. 加载所有图片和Masks (Load all images and masks)
        print(f"Loading images and masks from: {image_dir}")
        pixel_values, num_patches_list, image_mask = load_images_and_masks(image_dir)
        
        if pixel_values is None:
            print(f"Error: Failed to load any images from directory: {image_dir}")
            return

        pixel_values = pixel_values.to(self.device, dtype=self.model.dtype)
        image_mask = image_mask.to(self.device, dtype=self.model.dtype)
        print(f"Found {len(num_patches_list)} images and masks.")

        # 2. 读取任务描述 (Read the task description)
        try:
            with open(task_file_path, 'r', encoding='utf-8') as file:
                task_description = file.read()
        except FileNotFoundError:
            print(f"Error: Task file not found at {task_file_path}")
            return
        
        # 3. 构建问题 (Construct the question)
        question = '<image>\n' * len(num_patches_list) + task_description

        # 4. 执行推理 (Run inference)
        print("Running inference...")
        generation_config = dict(max_new_tokens=2048, do_sample=False)
        
        response = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config,
            history=None,
            return_history=False,
            # --- 新增参数 ---
            image_mask=image_mask,
            num_patches_list=num_patches_list
        )
        
        # 5. 打印结果 (Print the results)
        print("\n" + "="*20 + " Inference Result " + "="*20)
        print(f"User Question:\n{question}")
        print("-" * 50)
        print(f"Assistant Response:\n{response}")
        print("=" * 58 + "\n")


# --- 主程序入口 (Main Execution Block) ---

if __name__ == '__main__':
    # --- 请在这里配置您的路径 (Please configure your paths here) ---
    # 模型路径
    MODEL_PATH = '/pretrained/InternVL2-8B-llapa'
    
    # 示例文件夹路径 (请根据您的实际情况修改)
    SAMPLE_BASE_PATH = './sample' # 假设脚本与 'sample' 文件夹在同一目录下
    
    # 图片文件夹和任务文件路径
    IMAGE_DIR = os.path.join(SAMPLE_BASE_PATH, 'sample1')
    TASK_FILE = os.path.join(SAMPLE_BASE_PATH, 'task.txt') # 假设您的任务文件名是 task.txt

    # --- 检查路径是否存在 (Check if paths exist) ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path does not exist: {MODEL_PATH}")
    elif not os.path.exists(IMAGE_DIR):
        print(f"Error: Image directory does not exist: {IMAGE_DIR}")
    elif not os.path.exists(TASK_FILE):
        print(f"Error: Task file does not exist: {TASK_FILE}")
    else:
        # 初始化并运行推理
        infer_engine = SingleInstanceInfer(model_path=MODEL_PATH)
        infer_engine.predict(image_dir=IMAGE_DIR, task_file_path=TASK_FILE)
