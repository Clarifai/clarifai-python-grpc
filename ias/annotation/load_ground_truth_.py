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

GT_LABELS_ = [label + '_y' for label in GT_LABELS]


def load_from_csv(input_ids, csv_path, safe_gt):

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
                    if _clean_key(key) is not None:
                        # Catch exceptions to avoid errors related to incorrect ground truth entries
                        try:
                            if int(line[key]):
                                gt_labels.append(_clean_key(key))
                        except:
                            logging.warning('\t Incorrect ground truth for {}. Input ignored.'.format(line['video_id']))
                            break
                if not gt_labels:
                    gt_labels = [safe_gt]
                ground_truth[hash] = gt_labels
    
    # Count number of inputs with no ground truth
    no_gt_count = sum([1 for input_id in input_ids if not input_id in ground_truth]) 
    
    if no_gt_count > 0:
        logging.info("Ground truth was extracted from csv. {} inputs do not have ground truth.". format(no_gt_count)) 
    else:
        logging.info("Ground truth was extracted from csv for all inputs.")

    return ground_truth, no_gt_count
        

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

    return ground_truth, 0


def _clean_key(key):
    
    if key in GT_LABELS:
        return key
    elif key in GT_LABELS_:
        return key[:-2]
    else:
        return None