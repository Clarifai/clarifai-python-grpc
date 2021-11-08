import os
import json
import argparse
import itertools
import logging

# Import in the Clarifai gRPC based objects needed
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.json_format import MessageToDict
import load_ground_truth

# Setup logging
logging.basicConfig(format='%(asctime)s %(message)s \t')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Construct the communications channel and the object stub to call requests on.
channel = ClarifaiChannel.get_json_channel()
stub = service_pb2_grpc.V2Stub(channel)

def process_response(response):
    if response.status.code != status_code_pb2.SUCCESS:
        logger.error("There was an error with your request!")
        logger.error("\tDescription: {}".format(response.status.description))
        logger.error("\tDetails: {}".format(response.status.details))
        raise Exception("Request failed, status code: " + str(response.status.code))

def get_input_ids(metadata):
  ''' Get list of all inputs (ids of videos that were uploaded) from the app '''

  # Get inputs
  list_inputs_response = stub.ListInputs(
                         service_pb2.ListInputsRequest(page=1, per_page=1000),
                         metadata=metadata
  )
  process_response(list_inputs_response)

  # Extract input ids
  input_ids = {}
  for input_object in list_inputs_response.inputs:
    json_obj = MessageToDict(input_object)
    input_ids[json_obj['id']] = json_obj['data']['metadata']
  logger.info("Input ids fetched. Number of fetched inputs: {}".format(len(input_ids)))

  # # ------ DEBUG CODE
  # input_ids_ = {}
  # for id in list(input_ids.keys())[200:250]:
  #   input_ids_[id] = input_ids[id]
  # input_ids = input_ids_
  # print("Number of selected inputs: {}".format(len(input_ids)))
  # # ------ DEBUG CODE

  return input_ids, len(input_ids)


def get_ground_truth(args, input_ids):
  ''' Get list of ground truth concepts for every input id'''

  if os.path.exists(args.ground_truth):
    ground_truth, no_gt_count = load_ground_truth.load_from_csv(input_ids, args.ground_truth, args.safe_gt)
  else:
    ground_truth, no_gt_count = load_ground_truth.load_from_metadata(input_ids)

  # # ------ DEBUG CODE
  # # Compute list of unique labels
  # labels = list(itertools.chain(*[ground_truth[input_id] for input_id in input_ids]))
  # labels_count = {l:labels.count(l) for l in labels}
  # print("Ground truth labels are: ")
  # [print("\t{}: {}".format(k, v)) for k, v in labels_count.items()]
  # # ------ DEBUG CODE

  return ground_truth, no_gt_count


def remove_inputs_without_gt(input_ids, ground_truth):
  ''' Eliminate inputs that do not have any ground truth'''

  input_ids_with_gt = {}
  for input_id in input_ids:
    if input_id in ground_truth:
      input_ids_with_gt[input_id] = input_ids[input_id]

  return input_ids_with_gt


def get_annotations(metadata, input_ids):
  ''' Get list of annotations for every input id'''

  # Variable to check if number of annotations per page is sufficient
  annotation_nb_max = 0
  # Number of inputs with duplicated annotations
  duplicate_count = 0 
  duplicated_inputs = []

  annotations = {} # list of concepts
  annotations_meta = {} # store metadata

  # Get annotations for every input id
  for input_id in input_ids:
    list_annotations_response = stub.ListAnnotations(
                                service_pb2.ListAnnotationsRequest(
                                input_ids=[input_id], 
                                per_page=30,
                                list_all_annotations=True
                                ),
      metadata=metadata
    )
    process_response(list_annotations_response)
    # TODO: make requests in batches

    meta_ = []

    # Loop through all annotations
    for annotation_object in list_annotations_response.annotations:
      ao = MessageToDict(annotation_object)

      # Check for concepts in data but also time segments
      concepts = []
      if 'concepts' in ao['data'] and len(ao['data']['concepts'])>0:
        concepts.append(ao['data']['concepts'][0]['name']) # TODO: check if id always = name
      elif 'timeSegments' in ao['data'] and 'concepts' in ao['data']['timeSegments'][0]['data'] and \
           len(ao['data']['timeSegments'][0]['data'])>0:
        concepts.append(ao['data']['timeSegments'][0]['data']['concepts'][0]['name'])

      # Get meta about all found concepts besides duplicates
      for concept in concepts:
        meta_.append((concept, ao['userId']))

    # Remove duplicates from meta, transform it into dictionary (for further convenience) and store
    meta = set(meta_)
    meta = [{'concept': m[0], 'userId': m[1]} for m in meta]
    annotations_meta[input_id] = meta

    # Singal potential duplicates
    if len(meta_) > len(meta):
        duplicate_count += 1
        duplicated_inputs.append({'input_id': input_id, 
                                  'video_id': input_ids[input_id]['source-file-line'],  
                                  'duplicates': len(meta_) - len(meta)})

    # Extract concepts only
    annotations[input_id] = [m['concept'] for m in meta if '2-' in m['concept']]

    # Update max count variable
    annotation_nb_max = max(annotation_nb_max, len(list_annotations_response.annotations))

  logger.info("Annotations fetched")
  logger.info("\tMaximum number of annotation entries per input: {}".format(annotation_nb_max))
  logger.info("\tNumber of annotated inputs with duplicates: {}".format(duplicate_count))

  # # ------ DEBUG CODE
  # # Print duplicates
  # print('\nInput id - Video id - Duplicates')
  # for input in duplicated_inputs:
  #   print("{} - {} - {}".format(input['input_id'], input['video_id'], input['duplicates']))
  # print('\n')
  # # ------ DEBUG CODE

  # # ------ DEBUG CODE
  # # Compute list of unique labels
  # concepts = list(itertools.chain(*[annotations[input_id] for input_id in input_ids]))
  # concepts_count = {c:concepts.count(c) for c in concepts}
  # print("Annotation concepts are: ")
  # [print("\t{}: {}".format(k, v)) for k, v in concepts_count.items()]
  # # ------ DEBUG CODE

  return annotations, annotations_meta


def aggregate_annotations(args, input_ids, annotations):
  ''' Count the number of different annotation labels for every input id'''

  aggregated_annotations = {}
  not_annotated_count = 0

  for input_id in input_ids:
    if args.broad_consensus:
      # Abbreviate concepts to a common key is they contain that key (ex. 2-HB)
      aggregation = []
      for annotation in annotations[input_id]:
        if args.positive_annotation in annotation:
          aggregation.append(args.positive_annotation)
        else:
          aggregation.append(annotation)
      # Aggregate (count number of occurences of each concept)
      aggregation = {a:aggregation.count(a) for a in aggregation}
    else:
      # Aggregate without abbreviation 
      aggregation = {a:annotations[input_id].count(a) for a in annotations[input_id]}
      
    if not aggregation:
        not_annotated_count += 1
    else:
      aggregated_annotations[input_id] = aggregation

  logger.info("Annotations aggregated.")

  # TODO: store aggregated annotations
  return aggregated_annotations, not_annotated_count


def compute_consensus(args, input_ids, aggregated_annotations):
  ''' Compute consensus among annotations for every input id based on aggregation'''

  # Variable to count how many times no full consensus has been reached
  no_consensus_count = 0
  # Varibale to store ids for inputs that had conflicting annotations in consensus
  conflict_ids = []

  def consesus_fun(value):
    return True if value >= args.consensus_count else False

  consensus = {}
  for input_id in input_ids:
      if input_id in aggregated_annotations:
        # Compute consensus
        aa = aggregated_annotations[input_id]
        consensus_exists = {k:consesus_fun(v) for k, v in aa.items()}

        # If no consensus exists, set consensus to None
        if not True in consensus_exists.values():
          consensus[input_id] = None
          no_consensus_count += 1
          continue

        # If conflict between consenuses exist,
        # keep only positive consensus
        if args.positive_annotation in consensus_exists and \
           args.safe_annotation in consensus_exists and \
           consensus_exists[args.positive_annotation] and \
           consensus_exists[args.safe_annotation]:
            # Save only consensus for positive annotations
            consensus_exists.pop(args.safe_annotation)
            conflict_ids.append(input_id)
       
        # Store consensus
        consensus[input_id] = [concept for concept, exists in consensus_exists.items() if exists]

  logger.info("Consensus computed.")
  return consensus, no_consensus_count, conflict_ids


def compute_classes(args, input_ids, consensus, ground_truth):
  ''' Compute belonging of each input to the following classes: 
      Ground truth _GP_ (positive), _GN_ (negative), _GS_ (safe), and
      Labels _LP_ (positive), _LN_ (negative), _LS_ (safe) '''

  classes = {}
  for input_id in input_ids:
    classes_ = []

    # Ground truth first
    if args.positive_gt in ground_truth[input_id]:
      classes_.append('_GP_')
    elif args.safe_gt in ground_truth[input_id]:
      classes_.append('_GN_')
      classes_.append('_GS_')
    else:
      classes_.append('_GN_')

    # Labels
    if input_id in consensus and consensus[input_id] is not None:
      if args.positive_annotation in consensus[input_id]:
        classes_.append('_LP_')
      elif args.safe_annotation in consensus[input_id]:
        classes_.append('_LN_')
        classes_.append('_LS_')
      else:
        classes_.append('_LN_')  

    classes[input_id] = classes_

  logger.info("Classes computed.")
  return classes


def compute_totals(input_ids, classes):
  ''' Compute total number of inputs for each class '''

  totals = {'_GP_': 0, '_GN_': 0, '_GS_': 0, '_LP_': 0, '_LN_': 0, '_LS_': 0}

  for input_id in input_ids:
    if input_id in classes:
      if '_GP_' in classes[input_id]:
        totals['_GP_'] += 1
      if '_GN_' in classes[input_id]:
        totals['_GN_'] += 1
      if '_GS_' in classes[input_id]:
        totals['_GS_'] += 1
      if '_LP_' in classes[input_id]:
        totals['_LP_'] += 1
      if '_LN_' in classes[input_id]:
        totals['_LN_'] += 1
      if '_LS_' in classes[input_id]:
        totals['_LS_'] += 1

  logger.info("Totals computed.")
  return totals
  

def compute_metrics(input_ids, classes, totals):
  ''' Compute defined matrics '''

  markers = {}

  # Compute counts
  metric_counts = {'TP': 0, 'TN': 0, 'TS': 0, 'NFP': 0, 'SFP': 0, 'FN': 0, 'FS': 0}
  for input_id in input_ids: 
    if input_id in classes:
      # True Positive (positive annotation and positive ground truth)
      if '_LP_' in classes[input_id] and '_GP_' in classes[input_id]:
        metric_counts['TP'] += 1
        markers[input_id] = 'TP'
      # True Negative (negative annotation and negative ground truth)
      if '_LN_' in classes[input_id] and '_GN_' in classes[input_id]:
        metric_counts['TN'] += 1
        markers[input_id] = 'TN'
      # True Safe (safe annotation and safe ground truth)
      if '_LS_' in classes[input_id] and '_GS_' in classes[input_id]:
        metric_counts['TS'] += 1
        markers[input_id] = 'TS'
      # Negative False Positive (positive annotation and negative ground truth)
      if '_LP_' in classes[input_id] and '_GN_' in classes[input_id]:
        metric_counts['NFP'] += 1
        markers[input_id] = 'NFP'
      # Safe False Positive (positive annotation and safe ground truth)
      if '_LP_' in classes[input_id] and '_GS_' in classes[input_id]:
        metric_counts['SFP'] += 1
        markers[input_id] = 'SFP'
      # False Negative (negative annotation and positive ground truth)
      if '_LN_' in classes[input_id] and '_GP_' in classes[input_id]:
        metric_counts['FN'] += 1
        markers[input_id] = 'FN'
      # False Safe (safe annotation and positive ground truth)
      if '_LS_' in classes[input_id] and '_GP_' in classes[input_id]:
        metric_counts['FS'] += 1
        markers[input_id] = 'FS'
    
  # Compute rates
  metric_rates = {}
  metric_rates['TP'] = metric_counts['TP'] / totals['_GP_'] if metric_counts['TP'] != 0 else 0
  metric_rates['TN'] = metric_counts['TN'] / totals['_GN_'] if metric_counts['TN'] != 0 else 0
  metric_rates['TS'] = metric_counts['TS'] / totals['_GS_'] if metric_counts['TS'] != 0 else 0
  metric_rates['NFP'] = metric_counts['NFP'] / totals['_GN_'] if metric_counts['NFP'] != 0 else 0
  metric_rates['SFP'] = metric_counts['SFP'] / totals['_GS_'] if metric_counts['SFP'] != 0 else 0
  metric_rates['FN'] = metric_counts['FN'] / totals['_GP_'] if metric_counts['FN'] != 0 else 0
  metric_rates['FS'] = metric_counts['FS'] / totals['_GP_'] if metric_counts['FS'] != 0 else 0

  logger.info("Metrics computed.")
  return metric_counts, metric_rates, markers


def plot_results(input_count, no_gt_count, not_annotated_count, no_consensus_count,
                 metric_counts, metric_rates, totals):
    ''' Print results in the console '''
    
    print("\n*******************************************")
    print("--------------- Ground truth --------------\n")
    print("Not available: {}".format(no_gt_count))
    print("Positives: {} | Negatives: {} | Safe: {}".format(totals['_GP_'], totals['_GN_'], totals['_GS_']))
    print("\n------------------ Labels -----------------\n")
    print("Retrieved: {} | Kept: {}".format(input_count, input_count-no_gt_count))
    print("Not annotated: {} | No consensus: {}".format(not_annotated_count, no_consensus_count))
    print("Positives: {} | Negatives: {} | Safe: {}".format(totals['_LP_'], totals['_LN_'], totals['_LS_']))
    print("\n-------------- Metrics (rates) ------------\n")
    print("TP: \t{}/{}\t\t= {:.2f}".format(metric_counts['TP'], totals['_GP_'], metric_rates['TP']))
    print("TN: \t{}/{}\t\t= {:.2f}".format(metric_counts['TN'], totals['_GN_'], metric_rates['TN']))
    print("TS: \t{}/{}\t\t= {:.2f}".format(metric_counts['TS'], totals['_GS_'], metric_rates['TS']))
    print("NFP: \t{}/{}\t\t= {:.2f}".format(metric_counts['NFP'], totals['_GN_'], metric_rates['NFP']))
    print("SFP: \t{}/{}\t\t= {:.2f}".format(metric_counts['SFP'], totals['_GS_'], metric_rates['SFP']))
    print("FN: \t{}/{}\t\t= {:.2f}".format(metric_counts['FN'], totals['_GP_'], metric_rates['FN']))
    print("FS: \t{}/{}\t\t= {:.2f}".format(metric_counts['FS'], totals['_GP_'], metric_rates['FS']))
    print("*******************************************\n")
    

def get_false_annotations(input_ids, ground_truth, annotations_meta, consensus, markers):
  ''' Get information about inputs that were mislabelled '''

  false_annotations = {}
  for input_id in input_ids:
    if input_id in markers:
      if markers[input_id] == 'NFP' or markers[input_id] == 'SFP' or \
         markers[input_id] == 'FN' or markers[input_id] == 'FS':
        # Get initial information minus some fields
        input = input_ids[input_id]
        [input.pop(key) for key in ['results', 'source-file-line', 'group']]

        # Add information about results
        input['marker'] = markers[input_id]
        input['ground_truth'] = ground_truth[input_id]
        input['consensus'] = consensus[input_id]

        # Count different concepts
        concepts = [annotation['concept'] for annotation in annotations_meta[input_id]]
        input['annotations'] = {concept:concepts.count(concept) for concept in concepts}
        false_annotations[input_id] = input

        # Add raw information about annotations
        input['annotation_meta'] = annotations_meta[input_id]
  
  logger.info("False annotations extracted and stored.")

  return false_annotations
        

def get_conflicting_annotations(input_ids, conflict_ids, ground_truth, annotations_meta, consensus):
  ''' Get information about annotations with conflicting consensus'''

  conflicts = {}
  for input_id in conflict_ids:

    # Get initial information minus some fields
    input = input_ids[input_id]
    [input.pop(key) for key in ['results', 'source-file-line', 'group'] if key in input]

    # Add information about results
    input['ground_truth'] = ground_truth[input_id]
    input['consensus'] = consensus[input_id]

    # Count different concepts
    concepts = [annotation['concept'] for annotation in annotations_meta[input_id]]
    input['annotations'] = {concept:concepts.count(concept) for concept in concepts}
    conflicts[input_id] = input

    # Add raw information about annotations
    input['annotation_meta'] = annotations_meta[input_id]
  
  logger.info("Data about conflicting annotations is extracted and stored.")

  return conflicts


def save_data(args, to_save, data, name):
  ''' Dump provided data to a json file '''

  if to_save:
    with open("{}/{}_{}_{}.json".format(args.out_path, 
                                        args.app_name, 
                                        args.experiment_name.replace(' ', '-'),
                                        name), 'w') as f:
      json.dump(data, f)


def main(args, metadata):

  logger.info("----- Experiment {} - {} running -----".format(args.app_name, args.experiment_name))

  # Get input ids
  input_ids, input_count = get_input_ids(metadata)
  # Save metadata to re-use later if needed
  save_data(args, args.save_input_meta, input_ids, 'input_metadata')

  # Get ground truth labels for every input and eliminate those that do not have it
  ground_truth, no_gt_count = get_ground_truth(args, input_ids)
  input_ids = remove_inputs_without_gt(input_ids, ground_truth)

  # Get annotations for every id together with their aggregations
  annotations, annotations_meta = get_annotations(metadata, input_ids)
  aggregated_annotations, not_annotated_count = aggregate_annotations(args, input_ids, annotations)

  # Compute consensus
  consensus, no_consensus_count, conflict_ids = compute_consensus(args, input_ids, aggregated_annotations)

  # Compute results
  classes = compute_classes(args, input_ids, consensus, ground_truth)
  totals = compute_totals(input_ids, classes)
  metric_counts, metric_rates, markers = compute_metrics(input_ids, classes, totals)

  # Plot statistics using computed values
  plot_results(input_count, no_gt_count, not_annotated_count, no_consensus_count,
               metric_counts, metric_rates, totals) 

  # Get and save fails
  false_annotations = get_false_annotations(input_ids, ground_truth, annotations_meta, consensus, markers)
  save_data(args, args.save_false_annotations, false_annotations, 'false_annotations')
  if conflict_ids:
    conflicts = get_conflicting_annotations(input_ids, conflict_ids, ground_truth, annotations_meta, consensus)
    save_data(args, args.save_conflicts, conflicts, 'conflicts')
  else:
    logger.info("No conflicts in annotations. Nothing to dump.")


if __name__ == '__main__':  
  parser = argparse.ArgumentParser(description="Run tracking.")
  parser.add_argument('--app_name',
                      default='',
                      help="Name of the app in Clarifai UI.")
  parser.add_argument('--api_key',
                      default='',
                      help="API key to the required application.")                     
  parser.add_argument('--experiment', 
                      default=1, 
                      choices={1, 2, 3, 4},
                      type=int, 
                      help="Which experiment to analyize. Depends on the app.")
  parser.add_argument('--consensus_count',
                      default=3,
                      type=int,
                      help="How many of the same definition to require for consensus.")
  parser.add_argument('--broad_consensus',
                      default=True,
                      type=lambda x: (str(x).lower() == 'true'),
                      help="Attempt to allow for a broad consensus (i.e. multiple hate speech labels all pool to hate speech.")
  parser.add_argument('--ground_truth', 
                      default='', 
                      help="Path to csv file with ground truth.")                    
  parser.add_argument('--out_path', 
                      default='', 
                      help="Path to general output directory for this script.")
  parser.add_argument('--save_input_meta',
                      default=False,
                      type=lambda x: (str(x).lower() == 'true'),
                      help="Save input metadata in file or not.")
  parser.add_argument('--save_false_annotations',
                      default=True,
                      type=lambda x: (str(x).lower() == 'true'),
                      help="Save information about false annotations inputs in file or not.")
  parser.add_argument('--save_conflicts',
                      default=True,
                      type=lambda x: (str(x).lower() == 'true'),
                      help="Save information about annotations with conflicting consensus.")

  args = parser.parse_args()

  metadata = (('authorization', 'Key {}'.format(args.api_key)),)

  if args.experiment == 1:
    # Experiment 1
    args.experiment_name = 'Hate Speech'
    args.positive_gt = 'hate_speech'
    args.safe_gt = 'safe'
    args.positive_annotation = '2-HB'
    args.safe_annotation = '2-not-hate'

  elif args.experiment == 2:
    # Experiment 2
    args.experiment_name = 'AD'
    args.positive_gt = 'adult_&_explicit_sexual_content'
    args.safe_gt = 'safe'
    args.positive_annotation = '2-AD'
    args.safe_annotation = '2-none-of-the-above'

  elif args.experiment == 3:
    # Experiment 3
    args.experiment_name = 'OP'
    args.positive_gt = 'obscenity_&_profanity'
    args.safe_gt = 'safe'
    args.positive_annotation = '2-OP'
    args.safe_annotation = '2-none-of-the-above'

  elif args.experiment == 4:
    # Experiment 4
    args.experiment_name = 'ID'
    args.positive_gt = 'illegal_drugs/tobacco/e-cigarettes/vaping/alcohol'
    args.safe_gt = 'safe'
    args.positive_annotation = '2-ID'
    args.safe_annotation = '2-none-of-the-above'

  main(args, metadata)