if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', default=r"..\inputs\img_015_SRF_4_HR.png", help='image path')
    parser.add_argument('--layer_name', default=r"A3", help='Attention layer to combine')
    parser.add_argument('--heatmap_path', default=r".\Visual\\", help='path to the heat map')
    parser.add_argument('--save_path', default=r".\Combined", help='path to save the combined heat map')

    args = parser.parse_args()
    imgpath = args.img_path
    layer = args.layer_name
    heatmap_dir = args.heatmap_path
    save_dir = args.save_path
    import os
    import shutil
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.mkdir(save_dir)

    import cv2
    tmp = cv2.imread(heatmap_dir + layer + "_" + str(5000) + ".png")
    shape = tmp.shape
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (shape[1], shape[0]))

    for i in range(5000, 400000, 5000):
        heat_path = heatmap_dir + layer + "_" + str(i) + ".png"
        A = cv2.imread(heat_path)
        print(img.shape, A.shape)
        add_img = cv2.addWeighted(img, 0.3, A, 0.7, 0)
        cv2.imwrite(save_dir + r"\Combined_" + str(i) + ".png", add_img)
