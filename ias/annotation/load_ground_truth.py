import csv
import logging

GT_LABELS = ['adult_&_explicit_sexual_content',
             'arms_&_ammunition',
             'crime',
             'death,_injury_or_military_conflict',
             'online_piracy',
             'hate_speech',
             'obscenity_&_profanity',
             'illegal_drugs/tobacco/e-cigarettes/vaping/alcohol',
             'spam_or_harmful_content',
             'terrorism',
             'debated_sensitive_social_issue']

def load_from_csv(input_ids, csv_path, negative_concept):

    # Map video ids to their input hash
    id_to_hash = {}
    sfl_to_hash = {} # source-file-line
    for hash, value in input_ids.items():
        id_to_hash[value['id']] = hash
        sfl_to_hash[value['source-file-line']] = hash

    # Extract ground truth labels for every input present in the file
    ground_truth = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for line in reader:
            line = {k.lower(): v for k, v in line.items()}

            # Check if exists in ground truth file 
            hash = None
            if line['video_id'] in id_to_hash:
                hash = id_to_hash[line['video_id']]
            elif line['video_id'] in sfl_to_hash:
                hash = sfl_to_hash[line['video_id']]

            # If exists, extract gt keys
            if hash:
                gt_labels = []
                for key in line:
                    if key in GT_LABELS and int(line[key]):
                        gt_labels.append(key)
                if not gt_labels:
                    gt_labels = [negative_concept]
                ground_truth[hash] = gt_labels
    
    # In case some inputs are not in the file, create empty lists
    no_gt_count = 0
    for input_id in input_ids:
        if not input_id in ground_truth:
            ground_truth[input_id] = []
            # video_id = input_ids[input_id]['id']
            # stl = input_ids[input_id]['source-file-line']
            no_gt_count += 1
    
    if no_gt_count > 0:
        logging.info("Ground truth was extracted from csv. {} inputs do not have ground truth.". format(no_gt_count)) 
    else:
        logging.info("Ground truth was extracted from csv for all inputs.")

    return ground_truth
        

def load_from_metadata(input_ids):

    ground_truth = {}

    for input_id, input_val in input_ids.items():
        gt_labels = []
        # Extract all positive labels from results fields
        for key, val in input_val['results'].items():
            if val == True:
                gt_labels.append(key)
            ground_truth[input_id] = gt_labels

    logging.info('Ground truth extracted from metadata.')

    return ground_truth