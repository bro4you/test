import os
from tkinter import Tk, filedialog
from PIL import Image, ImageEnhance
import cv2
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor

# Определение класса DnCNN
class DnCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_layers=17):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Загрузка модели DnCNN
def load_dncnn_model(model_path="models/model_DnCNN_sigma=25_epoch_37.pth"):
    model = DnCNN()  # Создаем экземпляр модели
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Загружаем веса
    model.eval()  # Переключаем модель в режим оценки (inference)
    return model

# Функция для удаления шума с изображения
def denoise_image(image, model):
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        output_tensor = model(image_tensor).squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255.0
    return output_tensor.byte().numpy()

def process_image(filename, input_folder, output_folder, denoiser_model):
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Ошибка загрузки изображения: {img_path}. Проверьте путь и имя файла.")
        return

    # 1. Повышение разрешения
    img_upscaled = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)

    # 2. Устранение шумов
    img_denoised = denoise_image(img_upscaled, denoiser_model)

    # 3. Коррекция яркости и контраста
    pil_img = Image.fromarray(cv2.cvtColor(img_denoised, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_img)
    img_contrasted = enhancer.enhance(1.3)
    enhancer = ImageEnhance.Brightness(img_contrasted)
    img_final = enhancer.enhance(1.1)

    output_path = os.path.join(output_folder, filename)
    img_final.save(output_path)

def process_images(input_folder, output_folder, denoiser_model):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Список файлов изображений
    filenames = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    with ThreadPoolExecutor() as executor:
        # Обрабатываем изображения параллельно
        futures = [executor.submit(process_image, filename, input_folder, output_folder, denoiser_model) for filename in filenames]
        for future in futures:
            future.result()  # Дожидаемся завершения всех задач

    print("Обработка завершена!")

if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    print("Выберите папку с исходными изображениями.")
    input_folder = filedialog.askdirectory(title="Выберите папку с исходными изображениями")

    print("Выберите папку для сохранения улучшенных изображений.")
    output_folder = filedialog.askdirectory(title="Выберите папку для сохранения улучшенных изображений")

    if input_folder and output_folder:
        # Загружаем модель DnCNN
        denoiser_model = load_dncnn_model()
        
        # Запускаем процесс улучшения изображений
        process_images(input_folder, output_folder, denoiser_model)
    else:
        print("Не все папки выбраны. Программа завершена.")
