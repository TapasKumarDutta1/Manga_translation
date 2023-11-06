from torch import nn
import torch
import cv2
from matplotlib import pyplot as plt


class DETRModel(nn.Module):
    def __init__(self, num_classes=2, num_queries=100):
        super(DETRModel, self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.model = torch.hub.load(
            "facebookresearch/detr", "detr_resnet50", pretrained=True
        )
        self.in_features = self.model.class_embed.in_features

        self.model.class_embed = nn.Linear(
            in_features=self.in_features, out_features=self.num_classes
        )
        self.model.num_queries = self.num_queries

    def forward(self, images):
        return self.model(images)


def view_sample(model, device, img_path):
    img = cv2.imread(img_path) / 255
    img1 = img[:, :, ::-1]
    oh, ow, _ = img1.shape
    rh = oh / 512
    rw = ow / 512
    img = cv2.resize(img1, (512, 512))
    img = torch.tensor(img).permute(2, 0, 1)
    images = torch.unsqueeze(img, 0)
    _, h, w = images[0].shape  # for de normalizing images

    images = list(img.to(device) for img in images)

    sample = images[0].permute(1, 2, 0).cpu().numpy()

    model.eval()
    model = model.float().to(device)
    cpu_device = torch.device("cpu")
    with torch.no_grad():
        outputs = model(torch.unsqueeze(images[0], 0).float())

    outputs = [{k: v.to(cpu_device) for k, v in outputs.items()}]

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    oboxes = outputs[0]["pred_boxes"][0].detach().cpu().numpy()
    oboxes = oboxes * [512, 512, 512, 512]
    prob = outputs[0]["pred_logits"][0].softmax(1).detach().cpu().numpy()[:, 0]
    coords = {}
    for en, (box, p) in enumerate(zip(oboxes, prob)):
        cx, cy, h, w = box[0], box[1], box[2], box[3]
        if p > 0.5:
            x1, y1, x2, y2 = cx - h / 2, cy - w / 2, cx + h / 2, cy + w / 2
            x1, y1, x2, y2 = int(x1 * rw), int(y1 * rh), int(x2 * rw), int(y2 * rh)
            img = img1[y1:y2, x1:x2, :]
            print(en)
            cv2.imwrite("detected/" + str(en) + ".jpg", 255 * img)
            plt.imshow(img)
            plt.show()
            img1[y1:y2, x1:x2, :] = [255, 255, 255]
            coords[str(en)] = [y1, y2, x1, x2]
    cv2.imwrite("removed.jpg", img1 * 255)
    return coords


def detect(img_path):
    model = DETRModel()
    model.load_state_dict(torch.load("/content/detr_best_0 (1).pth"))
    return view_sample(model=model, device=torch.device("cpu"), img_path=img_path)

