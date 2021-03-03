import os
import sys

import cv2 as cv
import numpy as np
from settings import proj_dir

sys.path.append(os.getcwd())

DATA_FOLDER = os.path.join(proj_dir, 'data', 'tf-records-card-detect')
OUTPUT = os.path.join(proj_dir, 'data', 'dataset')
MAX_LEN = 640
MIN_LEN = 480

im_fns = os.listdir(os.path.join(DATA_FOLDER))
im_fns.sort()

if not os.path.exists(os.path.join(OUTPUT, "image")):
    os.makedirs(os.path.join(OUTPUT, "image"))
if not os.path.exists(os.path.join(OUTPUT, "label")):
    os.makedirs(os.path.join(OUTPUT, "label"))

for im_fn in im_fns:
    try:
        _, fn = os.path.split(im_fn)
        bfn, ext = os.path.splitext(fn)
        if ext.lower() not in ['.jpg', '.png']:
            continue

        gt_path = os.path.join(DATA_FOLDER, 'label-' + bfn + '.txt')
        img_path = os.path.join(DATA_FOLDER, im_fn)

        img = cv.imread(img_path)
        img_size = img.shape
        im_size_min = np.min(img_size[0:2])
        im_size_max = np.max(img_size[0:2])

        im_scale = float(MIN_LEN) / float(im_size_min)
        if np.round(im_scale * im_size_max) > MAX_LEN:
            im_scale = float(MAX_LEN) / float(im_size_max)
        new_h = int(img_size[0] * im_scale)
        new_w = int(img_size[1] * im_scale)

        new_h = new_h if new_h % 16 == 0 else (new_h // 16 + 1) * 16
        new_w = new_w if new_w % 16 == 0 else (new_w // 16 + 1) * 16

        re_im = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        re_size = re_im.shape

        polys = []
        with open(gt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            splitted_line = line.strip().lower().split(',')
            xmin, ymin, xmax, ymax = map(float, splitted_line[:8])
            xmin, xmax = [x / img_size[1] * re_size[1] for x in [xmin, xmax]]
            ymin, ymax = [y / img_size[0] * re_size[0] for y in [ymin, ymax]]
            polys.append([xmin, ymin, xmax, ymax])

            # cv.polylines(re_im, [poly.astype(np.int32).reshape((-1, 1, 2))], True,color=(0, 255, 0), thickness=2)

        cv.imwrite(os.path.join(OUTPUT, "image", fn), re_im)
        with open(os.path.join(OUTPUT, "label", bfn) + ".txt", "w") as f:
            for p in polys:
                line = ",".join(str(p[i]) for i in range(4))
                f.writelines(line + "\r\n")
                # for p in res_polys:
                #    cv.rectangle(re_im,(p[0],p[1]),(p[2],p[3]),color=(0,0,255),thickness=1)

                # cv.imshow("demo",re_im)
                # cv.waitKey(0)
    except Exception as err:
        print(err)
        go_on = input("Go on? >")
        if go_on != 'y':
            raise err
        print("Error processing {}".format(im_fn))
print('done.')
