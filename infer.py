import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from modules.model_architecture.UMT_PixelCNN import UMT_PixelCNN
from transformers import AutoTokenizer, RobertaConfig
from modules.resnet import resnet as resnet
from modules.resnet.resnet_utils import myResnet
import numpy as np
import json
# === Đường dẫn ===
output_dir = "/home/vms/bags/MNER-Vietnamese/tmp/train_umt_pixelcnn_fixedlr_2016_beta0.5_theta0.05_sigma0.005_lr3e-5_fixcrf"
model_config_path = os.path.join(output_dir, "model_config.json")
model_path = os.path.join(output_dir, "pytorch_model.bin")
encoder_path = os.path.join(output_dir, "pytorch_encoder.bin")

# === Tải cấu hình từ file ===
with open(model_config_path, "r", encoding="utf-8") as f:
    model_config = json.load(f)

bert_model_name = model_config["bert_model"]
max_seq_length = model_config["max_seq_length"]
num_labels = model_config["num_labels"]  # 13
do_lower = model_config["do_lower"]  # False
label_map_from_file = model_config["label_map"]  # dict: {"1": "B-ORG", ...}

# Xây dựng label_list theo index (1 → 12)
label_list = [""] * (num_labels - 1)  # vì num_labels = 13 → có 12 nhãn thực
for str_id, label in label_map_from_file.items():
    idx = int(str_id) - 1  # vì label_map bắt đầu từ 1, nhưng list bắt đầu từ 0
    label_list[idx] = label

# Sắp xếp lại theo thứ tự index (đảm bảo label_list[i] = nhãn có id = i+1)
# Ví dụ: label_list[0] = "B-ORG" (vì id=1), label_list[7] = "O" (vì id=8)
# Nhưng ta cần danh sách theo thứ tự id tăng dần → đã đúng do vòng lặp trên gán theo idx

# Tuy nhiên, để an toàn, ta tạo lại từ dict:
label_list = []
for i in range(1, num_labels):  # 1 đến 12
    label_list.append(label_map_from_file[str(i)])

# === Giả định auxlabel_list (bạn cần kiểm tra, nhưng thường là 6 hoặc 7) ===
# Dựa trên code train: auxnum_labels = len(auxlabel_list) + 1 = 7 → auxlabel_list có 6 phần tử
auxlabel_list = ["O", "B", "I", "X", "<s>", "</s>"]
auxnum_labels = len(auxlabel_list) + 1  # 7

# === Thiết lập device ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Khởi tạo mô hình ===
config = RobertaConfig.from_pretrained(bert_model_name, cache_dir='cache')
model = UMT_PixelCNN(config, num_labels_=num_labels, auxnum_labels=auxnum_labels)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Encoder ảnh ===
net = getattr(resnet, 'resnet152')()
encoder = myResnet(net, if_fine_tune=False, device=device)
encoder.load_state_dict(torch.load(encoder_path, map_location=device))
encoder.to(device)
encoder.eval()

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower)

# === Image transforms (giống training) ===
crop_size = 224
ti_crop_size = 32

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_for_ti = transforms.Compose([
    transforms.Resize([ti_crop_size, ti_crop_size]),
    transforms.ToTensor(),
    transforms.Normalize((0.48, 0.498, 0.531), (0.214, 0.207, 0.207))
])

def image_process(image_path, transform):
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # fallback to background
        bg_path = "/home/vms/bags/vlsp_all/origin+image/VLSP2016/ner_image/background.jpg"
        if not os.path.exists(bg_path):
            # tạo ảnh đen nếu không có background.jpg
            image = Image.new("RGB", (256, 256), (0, 0, 0))
        else:
            image = Image.open(bg_path).convert("RGB")
        return transform(image)

# === Hàm inference chính ===
def infer(text: str, image_path: str):
    # 1. Tiền xử lý ảnh
    img_feat = image_process(image_path, transform).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    img_ti_feat = image_process(image_path, transform_for_ti).unsqueeze(0).to(device)  # [1, 3, 32, 32]

    # 2. Tokenize như training: từng từ
    words = text.strip().split()
    tokens = []
    for word in words:
        tokenized = tokenizer.tokenize(word)
        tokens.extend(tokenized)

    # Cắt độ dài
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:max_seq_length - 2]

    # Thêm <s>, </s>
    ntokens = ["<s>"] + tokens + ["</s>"]
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)
    added_input_mask = [1] * (len(input_ids) + 49)  # như trong training

    # Padding
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        segment_ids.append(0)
        input_mask.append(0)
        added_input_mask.append(0)

    # Chuyển sang tensor
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(device)
    input_mask = torch.tensor([input_mask], dtype=torch.long).to(device)
    added_input_mask = torch.tensor([added_input_mask], dtype=torch.long).to(device)

    # 3. Trích xuất đặc trưng ảnh
    with torch.no_grad():
        imgs_f, img_mean, img_att = encoder(img_feat)

    # 4. Tạo trans_matrix (giống training)
    trans_matrix = np.zeros((auxnum_labels, num_labels), dtype=float)
    # Dùng nhánh else vì num_labels=13 <= 70
    trans_matrix[0,0] = 1   # pad → pad
    trans_matrix[1,1] = 1   # O → O
    trans_matrix[2,2] = 0.25  # B → B-MISC
    trans_matrix[2,4] = 0.25  # B → B-PER
    trans_matrix[2,6] = 0.25  # B → B-ORG
    trans_matrix[2,8] = 0.25  # B → B-LOC
    trans_matrix[3,3] = 0.25  # I → I-MISC
    trans_matrix[3,5] = 0.25  # I → I-PER
    trans_matrix[3,7] = 0.25  # I → I-ORG
    trans_matrix[3,9] = 0.25  # I → I-LOC
    trans_matrix[4,10] = 1    # X → X
    trans_matrix[5,11] = 1    # <s> → <s>
    trans_matrix[6,12] = 1    # </s> → </s>
    trans_matrix = torch.tensor(trans_matrix, dtype=torch.float).to(device)

    # 5. Chạy mô hình
    with torch.no_grad():
        predicted_ids = model(
            input_ids=input_ids,
            segment_ids=segment_ids,
            input_mask=input_mask,
            added_attention_mask=added_input_mask,
            visual_embeds_mean=imgs_f,
            visual_embeds_att=img_att,
            trans_matrix=trans_matrix,
            image_decode=img_ti_feat,  # lưu ý: ở đây truyền img_ti_feat, KHÔNG phải None
            alpha=0.5,
            beta=0.5,
            theta=0.05,
            sigma=0.005,
            temp=0.179,
            temp_lamb=0.7,
            labels=None,
            auxlabels=None
        )  # [1, seq_len]

    # 6. Giải mã
    pred_ids = predicted_ids[0]
    input_mask_cpu = input_mask[0]

    result_tokens = []
    result_labels = []

    # Bỏ <s> (index 0) và </s> (index cuối cùng có mask=1)
    for j in range(1, len(input_mask_cpu)):
        if input_mask_cpu[j] == 0:
            break
        if j >= len(ntokens):
            break
        token = ntokens[j]
        if token in ["<s>", "</s>"]:
            continue
        label_id = pred_ids[j]
        if label_id == 0:
            label = "O"
        else:
            label = label_list[label_id - 1]  # vì label_map bắt đầu từ 1
        if label not in ["X", "<s>", "</s>"]:
            result_tokens.append(token)
            result_labels.append(label)

    return list(zip(result_tokens, result_labels))

# === Ví dụ ===
if __name__ == "__main__":
    text = "Tổng giám đốc Nguyễn Hòa Bình của Viettel phát biểu tại Hà Nội."
    image_path = "/home/vms/bags/vlsp_all/origin+image/VLSP2016/ner_image/000001.jpg"
    # Nếu bạn không có background.jpg, có thể tạo trống hoặc bỏ qua

    try:
        entities = infer(text, image_path)
        print("Kết quả:")
        for tok, lab in entities:
            print(f"{tok}\t{lab}")
    except Exception as e:
        print("Lỗi:", e)