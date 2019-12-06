import numpy as np
import os
import glob
from PIL import Image, ImageFilter
import cv2
from shutil import copyfile


def find_coeffs(pa, pb):
    """Finds the points to apply a transformation between 2 points
    in an image.
    Arguments:
        pa:
        pb:
    Returns:
    """
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


plot_labels = False  # set as True for debugging purposes, plots the first image with bounding boxes
if plot_labels:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

# set datasets names and numerosities
datasets = [
    ("train", 400),
    ("val", 100)
]

# folder paths
background_folder_path = os.path.join(os.getcwd(), "bg_images")
logo_folder_path = os.path.join(os.getcwd(), "logo_images")
dataset_folder_path = os.path.join(os.getcwd(), "dataset")
additional_folder_path = os.path.join(os.getcwd(), "additional_imgs")
os.makedirs(dataset_folder_path, exist_ok=True)

# get filenames for backgrounds and logos
background_images_paths = glob.glob(os.path.join(background_folder_path, "*.jpg"))
logo_images_paths = glob.glob(os.path.join(logo_folder_path, "*.png"))

# set some parameters for data augmentation
n_classes = len(logo_images_paths)
max_objects_per_img = 3
bg_size = (800, 450)  # WxH
ar_range = (3.0, 5.0)  # logo aspect ratio range
h_range = (10, 100)  # height range
angle_range = (-45.0, 45.0)  # rotation angle range
shearing_range = (0, 50)
pf = 0.25  # perspective factor max range

# write the names file (classnames)
names = [os.path.basename(x).split(".")[0] + "\n" for x in logo_images_paths]
with open(os.path.join(dataset_folder_path, "logo.names"), "w") as f:
    f.writelines(names)

backup_folder = os.path.join(os.getcwd(), "backup")
os.makedirs(backup_folder, exist_ok=True)

for d_i, d in enumerate(datasets):
    set_name = d[0]
    set_number = d[1]
    print("Starting dataset " + set_name)
    # create dataset folders
    images_out_path = os.path.join(dataset_folder_path, "images", set_name)
    labels_out_path = os.path.join(dataset_folder_path, "labels", set_name)
    os.makedirs(images_out_path, exist_ok=True)
    os.makedirs(labels_out_path, exist_ok=True)
    # choose a random background for each image
    bg_inds = np.random.choice(len(background_images_paths), set_number)
    for count, i in enumerate(bg_inds):
        bg_path = background_images_paths[i]
        # load the background
        bg = Image.open(bg_path).resize(bg_size, resample=Image.BILINEAR)
        n_objs = np.random.randint(1, max_objects_per_img)
        boxes = []
        classes = []
        for j in range(n_objs):
            # choose a random logo (in our case there is only one)
            logo_id = np.random.choice(len(logo_images_paths))
            logo_path = logo_images_paths[logo_id]
            # vary the aspect ratio randomly
            random_ar = np.random.random() * (ar_range[1] - ar_range[0]) + ar_range[0]
            # choose a random size
            random_h = np.random.randint(h_range[0], h_range[1])
            random_w = int(random_h * random_ar)
            # load the image
            logo = Image.open(logo_path).resize((random_w, random_h), resample=Image.BILINEAR).convert("RGBA")
            # logo = logo.rotate(np.random.random() * (angle_range[1] - angle_range[0]) + angle_range[0]
            #                    , resample=Image.BILINEAR, expand=True)
            # if np.random.random() > 0.5:
            #     logo = logo.filter(ImageFilter.GaussianBlur())
            #
            # if np.random.random() > 0.5:
            #     w, h = logo.size
            #     from_points = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
            #     new_points = [(np.random.random() * w * pf, 0),
            #                   (w - np.random.random() * w * pf, 0),
            #                   (w - np.random.random() * w * pf, h - np.random.random() * h * pf),
            #                   (0, h - np.random.random() * h * pf)]
            #     coeffs = find_coeffs(new_points, from_points)
            #     logo = logo.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

            # random shearing
            if np.random.random() > 0.0:
                ofs_l = np.random.random() > 0.5
                w, h = logo.size
                maxshear = max(shearing_range[1], h)
                offset = np.random.random() * (maxshear - shearing_range[0]) + shearing_range[0]
                from_points = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
                new_points = [(0, offset if ofs_l else 0),
                              (w - 1, offset if not ofs_l else 0),
                              (w - 1, h - 1 + (offset if not ofs_l else 0)),
                              (0, h - 1 + (offset if ofs_l else 0))]
                coeffs = find_coeffs(new_points, from_points)
                logo = logo.transform((w, int(h + abs(offset))), Image.PERSPECTIVE, coeffs, Image.BILINEAR)

            xy = (np.random.randint(0, bg.size[0] - logo.size[0])
                  , np.random.randint(0, bg.size[1] - logo.size[1]))
            # paste the logo on the background
            bg.paste(logo, xy, logo)
            boxes.append((xy[0], xy[1], logo.size[0], logo.size[1]))
            classes.append(logo_id)

        # save the image in the images folder
        bg.save(os.path.join(images_out_path, "%05d.jpg" % count))
        # write down annotations in the labels folder
        with open(os.path.join(labels_out_path, "%05d.txt" % count), "w") as f:
            lines = []
            for j, b in enumerate(boxes):
                tmp_box = ((b[0] + b[2] / 2.0) / bg_size[0], (b[1] + b[3] / 2.0) / bg_size[1]
                                , b[2] / bg_size[0], b[3] / bg_size[1])
                lines.append(str(classes[j])
                             + " %1.6f %1.6f %1.6f %1.6f\n"
                             % tmp_box)
            f.writelines(lines)

        # if the debugging flag is true, show the image with bounding boxes
        if count == 0 and plot_labels:
            plt.imshow(bg)
            for b in boxes:
                plt.gca().add_patch(patches.Rectangle(b[:2], b[2], b[3], linewidth=1
                                                      , edgecolor='r', facecolor='none'))
            plt.show()

    # add additional hand labeled images to the training dataset
    if d_i == 0:
        add_imgs = glob.glob(os.path.join(additional_folder_path, "*.jpg"))
        add_labels = glob.glob(os.path.join(additional_folder_path, "*.txt"))
        for i in add_imgs:
            copyfile(i, os.path.join(images_out_path, os.path.basename(i).split(".")[0] + ".jpg"))
        for i in add_labels:
            copyfile(i, os.path.join(labels_out_path, os.path.basename(i).split(".")[0] + ".txt"))

    # write down the list of files for this dataset
    files = glob.glob(os.path.join(images_out_path, "*.jpg"))
    with open(os.path.join(dataset_folder_path, set_name + ".txt"), "w") as f:
        f.writelines([x + "\n" for x in files])

    print("Ending dataset " + set_name)

# write down dataset details
with open(os.path.join(dataset_folder_path, "logo.data"), "w") as f:
    f.write("classes=" + str(len(names))
            + "\ntrain=" + os.path.join(dataset_folder_path, datasets[0][0] + ".txt")
            + "\nvalid=" + os.path.join(dataset_folder_path, datasets[1][0] + ".txt")
            + "\nnames=" + os.path.join(dataset_folder_path, "logo.names")
            + "\nbackup=" + backup_folder
            + "\neval=coco"
            )

# extract frames from video and put them in the test dataset

video_folder_path = os.path.join(os.getcwd(), "test_video")
video_paths = glob.glob(os.path.join(video_folder_path, "*.mp4"))
images_out_path = os.path.join(dataset_folder_path, "images", "test")
os.makedirs(images_out_path, exist_ok=True)
for i, vp in enumerate(video_paths):
    vidcap = cv2.VideoCapture(vp)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(images_out_path, "video%02d_frame%05d.jpg" % (i, count)), image)
            count += 1
