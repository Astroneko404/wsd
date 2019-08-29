import json

if __name__ == '__main__':
    json_folder_path = '/home/luoz3/wsd_data_test/PE_50000_NO_M/'
    true_path = json_folder_path + 'instance_collection_true'
    pred_path = json_folder_path + 'instance_collection_pred'
    true_count_path = json_folder_path + 'instance_count_true.json'
    pred_count_path = json_folder_path + 'instance_count_pred.json'

    pred_freq_count = {}
    with open(pred_path) as pred_json_file:
        for line in pred_json_file:
            data = (json.loads(line))
            abbr = data['abbr']
            sense = data['sense_pred']

            if abbr not in pred_freq_count:
                pred_freq_count[abbr] = {}
            if sense not in pred_freq_count[abbr]:
                pred_freq_count[abbr][sense] = 0
            pred_freq_count[abbr][sense] += 1
    with open(pred_count_path, 'w+') as pred_json_out:
        json.dump(pred_freq_count, pred_json_out)

    true_freq_count = {}
    with open(true_path) as true_json_file:
        for line in true_json_file:
            data = (json.loads(line))
            abbr = data['abbr']
            sense = data['sense']

            if abbr not in true_freq_count:
                true_freq_count[abbr] = {}
            if sense not in true_freq_count[abbr]:
                true_freq_count[abbr][sense] = 0
            true_freq_count[abbr][sense] += 1
    with open(true_count_path, 'w+') as true_json_out:
        json.dump(true_freq_count, true_json_out)
