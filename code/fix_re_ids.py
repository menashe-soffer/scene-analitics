import numpy as np
import matplotlib.pyplot as plt
import pickle
import sklearn
import copy
import seaborn as sns

from paths import *


def check_similarity(id_1, id_2, id_vec, pairwise_dist):

    frames_1 = np.argwhere(id_1 == id_vec).flatten()
    frames_2 = np.argwhere(id_2 == id_vec).flatten()
    avg_dist = pairwise_dist[frames_1][:, frames_2].mean()
    #print('{} , {}  \t:  {}'.format(id_1, id_2, avg_dist))

    return avg_dist


def parse_person_detection(person_detection):

    num_ids = len(person_detection)
    num_frames = (max([len(p) for p in person_detection]))

    #raw_ids = np.zeros((num_frames, num_ids), dtype=int)
    dtct_conf = np.zeros((num_frames, num_ids))
    mask_wgt = np.zeros((num_frames, num_ids))
    id_list, feat_list, t_list, bbox_list, mask_wgt_list = [], [], [], [], []

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
                mask_wgt_list.append(person_detection[id][i_frame]['mask'].sum())
    id_vec, feat_vec, t_vec, bbox_vec, mask_wgt_vec = np.array(id_list), np.array(feat_list), np.array(t_list), np.array(bbox_list).squeeze(), np.array(mask_wgt_list)


    return num_ids, num_frames, dtct_array, dtct_conf, mask_wgt_vec, id_vec, feat_vec, t_vec, bbox_vec



def fix_detctions(input_fname, output_fname, show=True):

    with open(input_fname, 'rb') as fd:
        person_detection = pickle.load(fd)['person_detection']
    # num_ids = len(person_detection)
    # num_frames = (max([len(p) for p in person_detection]))
    #
    # #raw_ids = np.zeros((num_frames, num_ids), dtype=int)
    # dtct_conf = np.zeros((num_frames, num_ids))
    # mask_wgt = np.zeros((num_frames, num_ids))
    # id_list, feat_list, t_list, bbox_list = [], [], [], []
    #
    # dtct_array = np.zeros((num_frames, num_ids), dtype=bool)
    # for id in range(1, num_ids):
    #     for i_frame in range(len(person_detection[id])):
    #         if len(person_detection[id][i_frame]) > 0:
    #             dtct_array[i_frame, id] = True
    #             dtct_conf[i_frame, id] = person_detection[id][i_frame]['conf']
    #             mask_wgt[i_frame, id] = person_detection[id][i_frame]['mask'].sum()
    #             id_list.append(id)
    #             feat_list.append(person_detection[id][i_frame]['feat'])
    #             t_list.append(i_frame)
    #             bbox_list.append(person_detection[id][i_frame]['bbox'])
    # id_vec, feat_vec, t_vec, bbox_vec = np.array(id_list), np.array(feat_list), np.array(t_list), np.array(bbox_list).squeeze()

    num_ids, num_frames, dtct_array, _, _, id_vec, feat_vec, _, _ = parse_person_detection(person_detection)

    # pairwize distances
    A_sqr = (feat_vec ** 2).sum(axis=1)
    pairwise_dist = A_sqr[:, np.newaxis] - 2 * feat_vec @ feat_vec.T + A_sqr[np.newaxis, :]


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
        inter_id_distances[idx, idx] = check_similarity(id, id, id_vec, pairwise_dist)
    for idx1, id in enumerate(id_list):
        frames = np.argwhere(dtct_array[:, id]).flatten().astype(int)
        candidates, scores = [], []
        for idx2, cand_id in enumerate(id_list):
            if dtct_array[frames, cand_id].sum() == 0:
                candidates.append(cand_id)
                inter_id_distances[idx1, idx2] = check_similarity(id, cand_id, id_vec, pairwise_dist)
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
    if show:
        fig, ax = plt.subplots(1, 1)
        sns.heatmap(np.round(inter_id_distances, decimals=2), ax=ax, annot=True, xticklabels=id_list, yticklabels=id_list)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(pairwise_dist)
        plt.show()



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

    return new_person_detection




if __name__ == '__main__':

    first_id, last_id = np.zeros(5, dtype=int), np.zeros(5, dtype=int)
    for i in range(1, 5):
        input_fnane = os.path.join(DATA_FOLDER, str(i))
        output_fname = input_fnane + '_'
        person_detection = fix_detctions(input_fname=input_fnane, output_fname=output_fname, show=False)
        if i == 1:
            all_person_detections = copy.copy(person_detection)
            first_id[i], last_id[i] = 1, len(all_person_detections) - 1
        else:
            first_id[i] = len(all_person_detections) + 1
            all_person_detections = all_person_detections + copy.copy(person_detection)
            last_id[i] = len(all_person_detections) - 1
    for i in range(1, 5):
        print('clip {} first id = {} last id = {}'.format(i, first_id[i], last_id[i]))

    num_ids, num_frames, dtct_array, dtct_conf, mask_wgt_vec, id_vec, feat_vec, _, _ = parse_person_detection(all_person_detections)
    # #
    # pca = sklearn.decomposition.PCA(n_components=100)
    # feat_vec = pca.fit_transform(feat_vec)
    # #
    # pairwize distances
    A_sqr = (feat_vec ** 2).sum(axis=1)
    pairwise_dist = A_sqr[:, np.newaxis] - 2 * feat_vec @ feat_vec.T + A_sqr[np.newaxis, :]
    fig, ax = plt.subplots(1, 1)
    ax.imshow(pairwise_dist)
    plt.show()

    cross_dist_mat, rel_dist_mat, mod_dist_mat = np.zeros((num_ids, num_ids)), np.zeros((num_ids, num_ids)), np.zeros((num_ids, num_ids))
    for i_clip in range(1, 4):
        # look for similarities between persons in clip1 and all other clips
        for id1 in range(first_id[i_clip], last_id[i_clip] + 1):
            for id2 in range(last_id[i_clip] + 2, len(all_person_detections)):
                frame_idxs_1 = np.argwhere(id_vec == id1).flatten().astype(int)
                frame_idxs_2 = np.argwhere(id_vec == id2).flatten().astype(int)
                if min(frame_idxs_1.size, frame_idxs_2.size) > 0:
                    #print('calculating distance id({}) and id({})'.format(id1, id2))
                    frame_idxs_1 = frame_idxs_1[np.argsort(mask_wgt_vec[frame_idxs_1])[-1000:]]
                    frame_idxs_2 = frame_idxs_2[np.argsort(mask_wgt_vec[frame_idxs_2])[-1000:]]
                    cross_dist = pairwise_dist[frame_idxs_1][:, frame_idxs_2].mean()
                    relative_cross_dist = cross_dist / np.sqrt(pairwise_dist[frame_idxs_1][:, frame_idxs_1].mean() * pairwise_dist[frame_idxs_2][:, frame_idxs_2].mean())
                    modified_dist = ((feat_vec[frame_idxs_1].mean(axis=0) - feat_vec[frame_idxs_2].mean(axis=0)) ** 2).sum()
                    #print('\tdist={:5.3f}  relative={:5.3f}'.format(cross_dist, relative_cross_dist))
                    cross_dist_mat[id1, id2] = cross_dist
                    rel_dist_mat[id1, id2] = relative_cross_dist
                    mod_dist_mat[id1, id2] = modified_dist

    fig, ax = plt.subplots(1, 1)
    sns.heatmap(np.round(cross_dist_mat, decimals=2), ax=ax, annot=True)#, xticklabels=id_list, yticklabels=id_list))
    fig, ax = plt.subplots(1, 1)
    sns.heatmap(np.round(rel_dist_mat, decimals=2), ax=ax, annot=True)#, xticklabels=id_list, yticklabels=id_list))
    fig, ax = plt.subplots(1, 1)
    sns.heatmap(np.round(mod_dist_mat, decimals=2), ax=ax, annot=True)#, xticklabels=id_list, yticklabels=id_list))
    fig, ax = plt.subplots(1, 1)
    sns.heatmap((cross_dist_mat > 0) * (rel_dist_mat < 1.5), ax=ax, annot=True)#, xticklabels=id_list, yticklabels=id_list))
    plt.show()

    # decide on re-ID's
    modifications = dict({2: dict(), 3: dict(), 4: dict()})
    for clip_1 in range(1, 4):
        for clip_2 in range(clip_1 + 1, 5):
            for id1 in range(first_id[clip_1], last_id[clip_1] + 1):
                id2 = first_id[clip_2] + np.argmin(rel_dist_mat[id1, first_id[clip_2]:last_id[clip_2] + 1])
                if rel_dist_mat[id1, id2] < 1.5:
                    new_id = id1
                    if clip_1 > 1:
                        for key in modifications[clip_1]:
                            if key == id1:
                                new_id = modifications[clip_1][key]
                    modifications[clip_2][id2] = new_id
                    print('in clip {}  modify id={} to id-{}'.format(clip_2, id2, new_id))



    # now decompose person_data_all again applying the modifications
    for i_clip in range(1, 5):
        person_detection = copy.copy(all_person_detections)
        for id in range(len(all_person_detections)):
            if (id < first_id[i_clip]) or (id > last_id[i_clip]):
                person_detection[id] = []
        if i_clip > 1:
            for key in modifications[i_clip]:
                person_detection[modifications[i_clip][key]] = person_detection[key]
                person_detection[key] =[]
        ouput_fname = os.path.join(DATA_FOLDER, str(i_clip)) + '_'
        with open(ouput_fname, 'wb') as fd:
            pickle.dump(dict({'person_detection': person_detection}), fd)







