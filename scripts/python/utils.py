import sqlite3
import cv2
import numpy as np
import argparse

MAX_IMAGE_ID = 2**31 - 1
def read_KRT():
    intrinsics = {}
    extrinsics = {}

    with open('/home/gaini/capstone/dataset/KRT') as f:
        lines = f.readlines()

        num = len(lines)
        i = 0
        while i < num:
            camera_id = int(lines[i])
            a = lines[i+1].split(" ")
            b = lines[i+2].split(" ")

            aa = a[2]
            bb = b[2]
            K = [[0, 0, 0], [0, 0, 0], [0, 0, 1]]
            K[0][0] = a[0]
            K[1][1] = b[1]
            K[0][2] = aa[:-2]
            K[1][2] = bb[:-2]
            intrinsics[camera_id] = np.array(K).astype('float64')


            extr = np.zeros((3, 4))
            extr1 = lines[i + 5].split(" ")
            extr2 = lines[i + 6].split(" ")
            extr3 = lines[i + 7].split(" ")
            extr[0] = extr1
            extr[1] = extr2
            extr[2] = extr3
            extrinsics[camera_id] = extr.astype('float64')

            i += 9

    return intrinsics, extrinsics

def get_keypoints(cursor, image_id):
    cursor.execute("SELECT * FROM keypoints WHERE image_id = ?;", (image_id,))
    image_idx, n_rows, n_columns, raw_data = cursor.fetchone()
    kypnts = np.frombuffer(raw_data, dtype=np.float32).reshape(n_rows, n_columns).copy()
    kypnts = kypnts[:,0:2]
    return kypnts

def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2

def vizualize_intrinsics_extrinsics(F, id1, id2):
    # pts1_sample = match_positions[:, :2][20:30, :]
    pts1_sample = np.array([ [800, 300], [400, 1000]])

    num_points = pts1_sample.shape[0]

    epilines = cv2.computeCorrespondEpilines(pts1_sample.reshape(-1, 1, 2), 1, F)

    img2 = cv2.imread('/home/gaini/capstone/dataset/frames/' + id2 + '/' + id2 + '_038512.png')
    img1 = cv2.imread('/home/gaini/capstone/dataset/frames/' + id1 + '/' + id1 + '_038512.png')
    img1_epilines = img1.copy()
    img2_epilines = img2.copy()
    for i in range(num_points):
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # line_color = tuple(np.random.randint(0, 255, size=3).tolist())

        # Draw point in image 1
        img1_epilines = cv2.drawMarker(img1_epilines, tuple(pts1_sample[i, :].astype(np.int32)), color,
                                       markerType=cv2.MARKER_CROSS, thickness=2)

        a, b, c = epilines[i][0]
        x0, y0 = 0, int(-c / b)
        x1, y1 = img2.shape[1], int(-(c + a * img2.shape[1]) / b)
        img2_epilines = cv2.line(img2_epilines, (x0, y0), (x1, y1), color, 1)

    # Display images with epilines and points
    cv2.namedWindow('Image 1 with points', cv2.WINDOW_NORMAL)

    cv2.imshow('Image 1 with points', img1_epilines)

    cv2.namedWindow('Image 2 with epilines', cv2.WINDOW_NORMAL)
    cv2.imshow('Image 2 with epilines', img2_epilines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def check():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    parser.add_argument("--outdir", default="./calculated_extrinsics.txt")
    args = parser.parse_args()

    connection = sqlite3.connect(args.database_path)
    cursor = connection.cursor()

    list_image_ids = []
    img_ids_to_names_dict = {}
    cursor.execute(
        'SELECT image_id, name, cameras.width, cameras.height FROM images LEFT JOIN cameras ON images.camera_id == cameras.camera_id;')
    for row in cursor:
        image_idx, name, width, height = row
        list_image_ids.append(image_idx)
        img_ids_to_names_dict[image_idx] = name

    cursor.execute('SELECT pair_id, rows, cols, data FROM two_view_geometries;')
    all_matches = {}
    for row in cursor:
        pair_id = row[0]
        rows = row[1]
        cols = row[2]
        raw_data = row[3]
        if (rows < 5):
            continue

        matches = np.frombuffer(raw_data, dtype=np.uint32).reshape(rows, cols)

        if matches.shape[0] < 5:
            continue

        all_matches[pair_id] = matches
    intrinsics, extrinsics = read_KRT()

    for key in all_matches:
        pair_id = key
        matches = all_matches[key]
        id1, id2 = pair_id_to_image_ids(pair_id)

        image_name1 = img_ids_to_names_dict[id1]
        image_name2 = img_ids_to_names_dict[id2]

        cam1 = image_name1.split('.')[0].split("_")[0]
        cam2 = image_name2.split('.')[0].split("_")[0]
        if  cam1 == "400029" and cam2 == "400019":
            keys1 = get_keypoints(cursor, id1)
            keys2 = get_keypoints(cursor, id2)

            match_positions = np.empty([matches.shape[0], 4])
            for i in range(0, matches.shape[0]):
                match_positions[i, :] = np.array(
                    [keys1[matches[i, 0]][0], keys1[matches[i, 0]][1], keys2[matches[i, 1]][0],
                     keys2[matches[i, 1]][1]])

            F, mask = cv2.findFundamentalMat(match_positions[:, :2], match_positions[:, 2:4], cv2.FM_RANSAC, 0.1, 0.9)
            print("F", F)

            U, D, Vt = np.linalg.svd(F)
            print("D", D)

            str_id1 = image_name1.split('.')[0].split("_")[0]
            str_id2 = image_name2.split('.')[0].split("_")[0]
            K1 = intrinsics[int(str_id1)] @ intrinsics[int(str_id1)].T
            K2 = intrinsics[int(str_id2)] @ intrinsics[int(str_id2)].T

            r = D[0]
            s = D[1]

            u1 = U[:, 0]
            u2 = U[:, 1]
            u3 = U[:, 2]

            v1 = Vt[:, 0]
            v2 = Vt[:, 1]
            v3 = Vt[:, 2]
            left = (-1 * (r ** 2) * (v1.T @ (K1 @ v1))) * (u2.T @ (K2 @ u1))
            right = (r * s * (v1.T @ (K1 @ v2))) * (u2.T @ (K2 @ u2))

            # shoudl be zero
            print("diff", left-right)

            vizualize_intrinsics_extrinsics(F, cam1, cam2)


    cursor.close()
    connection.close()


if __name__ == "__main__":
    check()