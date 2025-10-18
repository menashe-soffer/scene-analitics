import numpy as np
import matplotlib.pyplot as plt
import pickle
#import sklearn
import copy
import seaborn as sns

from paths import *


def check_similarity(id_1, id_2, id_vec, pairwise_dist, pairwise_iou):

    frames_1 = np.argwhere(id_1 == id_vec).flatten()
    frames_2 = np.argwhere(id_2 == id_vec).flatten()
    avg_dist = pairwise_dist[frames_1][:, frames_2].mean()
    avg_iou = pairwise_iou[frames_1][:, frames_2].mean()
    #print('{} , {}  \t:  {}, {}'.format(id_1, id_2, avg_dist, avg_iou))

    return avg_dist


def fix_detctions(input_fname, output_fname):

    with open(input_fname, 'rb') as fd:
        person_detection = pickle.load(fd)['person_detection']
    num_ids = len(person_detection)
    num_frames = (max([len(p) for p in person_detection]))

    #raw_ids = np.zeros((num_frames, num_ids), dtype=int)
    dtct_conf = np.zeros((num_frames, num_ids))
    mask_wgt = np.zeros((num_frames, num_ids))
    id_list, feat_list, t_list, bbox_list = [], [], [], []

    dtct_array = np.zeros((num_frames, num_ids), dtype=bool)
    for id in range(1, num_ids):
        for i_frame in range(len(person_detection[id])):
            if len(person_detection[id][i_frame]) > 0:
                dtct_array[i_frame, id] = True
                dtct_conf[i_frame, id] = person_detection[id][i_frame]['conf']
                mask_wgt[i_frame, id] = person_detection[id][i_frame]['mask'].sum()
                id_list.append(id)
                feat_list.append(person_detection[id][i_frame]['feat'])
                t_list.append(i_frame)
                bbox_list.append(person_detection[id][i_frame]['bbox'])
    id_vec, feat_vec, t_vec, bbox_vec = np.array(id_list), np.array(feat_list), np.array(t_list), np.array(bbox_list).squeeze()


    # pairwize distances
    A_sqr = (feat_vec ** 2).sum(axis=1)
    pairwise_dist = A_sqr[:, np.newaxis] - 2 * feat_vec @ feat_vec.T + A_sqr[np.newaxis, :]
    # pairwize IOU
    dx = bbox_vec[:, 2] - bbox_vec[:, 0]
    dy = bbox_vec[:, 3] - bbox_vec[:, 1]
    bbox_area = dx * dy
    isct_x1 = np.maximum(bbox_vec[:, 0:1], bbox_vec[:, 0:1].T)
    isct_x2 = np.minimum(bbox_vec[:, 2:3], bbox_vec[:, 2:3].T)
    isct_y1 = np.maximum(bbox_vec[:, 1:2], bbox_vec[:, 1:2].T)
    isct_y2 = np.maximum(bbox_vec[:, 3:4], bbox_vec[:, 3:4].T)
    isct_dx = np.maximum(isct_x2 - isct_x1, 0)
    isct_dy = np.maximum(isct_y2 - isct_y1, 0)
    iou = (isct_dx * isct_dy) / (bbox_area[:, np.newaxis] + bbox_area[np.newaxis, :])




    id_list, frm_cnt_list = [], []
    for id in range(num_ids):
        frm_cnt = dtct_array[:, id].sum()
        if frm_cnt > 0:
            print('ID {}\t:\t {} frames'.format(id, frm_cnt))
            id_list.append(id)
            frm_cnt_list.append(frm_cnt)
    reord = np.argsort(frm_cnt_list)
    id_list, frm_cnt_list = np.array(id_list)[reord], np.array(frm_cnt_list)[reord]

    inter_id_distances = np.ones((len(id_list), len(id_list)))
    best_cross_distance = 1
    for idx, id in enumerate(id_list):
        inter_id_distances[idx, idx] = check_similarity(id, id, id_vec, pairwise_dist, iou)
    for idx1, id in enumerate(id_list):
        frames = np.argwhere(dtct_array[:, id]).flatten().astype(int)
        candidates, scores = [], []
        for idx2, cand_id in enumerate(id_list):
            if dtct_array[frames, cand_id].sum() == 0:
                candidates.append(cand_id)
                inter_id_distances[idx1, idx2] = check_similarity(id, cand_id, id_vec, pairwise_dist, iou)
                if inter_id_distances[idx1, idx2] < best_cross_distance:
                    best_cross_distance = inter_id_distances[idx1, idx2]
                    best_relative_cross = best_cross_distance / (np.sqrt(inter_id_distances[idx1, idx1] * inter_id_distances[idx2, idx2]) + 1e-16)
                    best_cross_1, best_cross_2 = id, cand_id
        if len(candidates) > 0:
            print('ID {} can also be:'.format(id), candidates)
    print('best match: {} - {} , distance: {:5.3f}, relative: {:5.3f}'.format(best_cross_1, best_cross_2, best_cross_distance, best_relative_cross))

    # inter_id_distances = np.ones((len(id_list), len(id_list)))
    # for idx1, id_1 in enumerate(id_list):
    #     for idx2, id_2 in enumerate(id_list):
    #         inter_id_distances[idx1, idx2] = check_similarity(id_1, id_2, id_vec, pairwise_dist, iou)
    fig, ax = plt.subplots(1, 1)
    sns.heatmap(np.round(inter_id_distances, decimals=2), ax=ax, annot=True, xticklabels=id_list, yticklabels=id_list)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(pairwise_dist)
    ax[1].imshow(iou)
    plt.show()

    # model = sklearn.manifold.TSNE(init='random')
    # embd = model.fit_transform(np.array(feat_list))
    # for id in range(1, num_ids):
    #     plt.scatter(feat_list[id_list==id, 0], feat_list[id_list==id, 1])
    # plt.show()


    # reassign ID's
    if (best_cross_distance < 0.5) and (best_relative_cross < 1.5):
        #from_id, to_id = best_cross_1, best_cross_2
        from_frames = np.argwhere([len(f) > 0 for f in person_detection[best_cross_1]]).flatten().astype(int)
        for i_frame in from_frames:
            person_detection[best_cross_2][i_frame] = copy.copy(person_detection[best_cross_1][i_frame])
        person_detection[best_cross_1] = []

    # erase very short detection
    if np.min(frm_cnt_list) <= 10:
        id_to_erase = id_list[np.argwhere(frm_cnt_list <= 10).flatten().astype(int)]
        for id in id_to_erase:
            person_detection[id] = []

    # get rid of empty ID's
    new_person_detection = [[]]
    for person in person_detection:
        if len(person) > 0:
            new_person_detection.append(person)

    with open(output_fname, 'wb') as fd:
        pickle.dump(dict({'person_detection': new_person_detection}), fd)




if __name__ == '__main__':

    for i in range(1, 5):
        input_fnane = os.path.join(DATA_FOLDER, str(i))
        output_fname = input_fnane + '_'
        fix_detctions(input_fname=input_fnane, output_fname=output_fname)



