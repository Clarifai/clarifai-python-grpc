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

def get_input_ids(args, metadata):
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
  # for id in list(input_ids.keys())[0:100]:
  #   input_ids_[id] = input_ids[id]
  # input_ids = input_ids_
  # logger.info("Number of selected inputs: {}".format(len(input_ids)))
  # # ------ DEBUG CODE

  return input_ids, len(input_ids)


def get_ground_truth(args, input_ids):
  ''' Get list of ground truth concepts for every input id'''

  if os.path.exists(args.ground_truth):
    ground_truth, no_gt_count = load_ground_truth.load_from_csv(input_ids, args.ground_truth, args.negative_concept)
  else:
    ground_truth, no_gt_count = load_ground_truth.load_from_metadata(input_ids)

  # Count the number of positive and negative labels
  positive_count, negative_count = 0, 0
  for input_id in input_ids:
    if input_id in ground_truth:
      if args.positive_gt_label in ground_truth[input_id]:
        positive_count += 1
      if ground_truth[input_id] == [args.negative_concept]:
        negative_count += 1

  # # ------ DEBUG CODE
  # # Compute list of unique labels
  # labels = list(itertools.chain(*[ground_truth[input_id] for input_id in input_ids]))
  # labels_count = {l:labels.count(l) for l in labels}
  # logger.info("Ground truth labels are: ")
  # [logger.info("\t{}: {}".format(k, v)) for k, v in labels_count.items()]
  # # ------ DEBUG CODE

  return ground_truth, positive_count, negative_count, no_gt_count


def remove_inputs_without_gt(input_ids, ground_truth):
  ''' Eliminate inputs that do not have any ground truth'''

  input_ids_with_gt = {}
  for input_id in input_ids:
    if input_id in ground_truth:
      input_ids_with_gt[input_id] = input_ids[input_id]

  return input_ids_with_gt


def get_annotations(args, metadata, input_ids):
  ''' Get list of annotations for every input id'''

  # Variable to check if number of annotations per page is sufficient
  annotation_nb_max = 0

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

    # Store annotations
    annotation = []
    annotation_meta = []
    for annotation_object in list_annotations_response.annotations:
      ao = MessageToDict(annotation_object)

      # Check for concepts in data but also time segments
      concepts = []
      if 'concepts' in ao['data'] and len(ao['data']['concepts'])>0:
        concepts.append(ao['data']['concepts'][0]['name']) # TODO: check if id always = name
      elif 'timeSegments' in ao['data'] and 'concepts' in ao['data']['timeSegments'][0]['data'] and \
           len(ao['data']['timeSegments'][0]['data'])>0:
        concepts.append(ao['data']['timeSegments'][0]['data']['concepts'][0]['name'])

      # Store all found concepts
      for concept in concepts:
        if not args.label_2_only:
          annotation.append(concept)
        elif '2-' in concept:
            annotation.append(concept)
        # Store metadata
        meta = {'concept': concept, 'userId': ao['userId']}
        annotation_meta.append(meta)  
        
    annotations[input_id] = annotation
    annotations_meta[input_id] = annotation_meta

    # Update max count variable
    annotation_nb_max = max(annotation_nb_max, len(list_annotations_response.annotations))

  logger.info("Annotations fetched. Maximum number of annotation entries per input: {}".format(annotation_nb_max))

  # # ------ DEBUG CODE
  # # Compute list of unique labels
  # concepts = list(itertools.chain(*[annotations[input_id] for input_id in input_ids]))
  # concepts_count = {c:concepts.count(c) for c in concepts}
  # logger.info("Annotation concepts are: ")
  # [logger.info("\t{}: {}".format(k, v)) for k, v in concepts_count.items()]
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
        if args.positive_concept_key in annotation:
          aggregation.append(args.positive_concept_key)
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

        # Extract all positive concepts with consensus
        positive_concepts_concensus = []
        for concept, exists in consensus_exists.items():
          if (args.positive_concept_key in concept) and exists:
            positive_concepts_concensus.append(concept)

        # Store all positive concepts with consensus, 
        # If no exists, chose negative concept as consensus
        if positive_concepts_concensus:
          consensus[input_id] = positive_concepts_concensus
        else:
          consensus[input_id] = [args.negative_concept]

  logger.info("Consensus computed.")
  return consensus, no_consensus_count


def compute_stats(args, input_ids, consensus, ground_truth):
  ''' Compute how many annotations match ground truth '''
  
  TP_count, TN_count, FP_count, FN_count = 0, 0, 0, 0

  stats = {}
  for input_id in input_ids:
    # Check first if this input was annotated (consensus varible contains only annotated inputs)
    # and check if ground truth exists for this input
    if input_id in consensus and input_id in ground_truth:
      # Check if consensus was reached
      if consensus[input_id] is not None: 
        if any([True for concept in consensus[input_id] if args.positive_concept_key in concept]): # annotation is positive
          if args.positive_gt_label in ground_truth[input_id]: # ground truth is positive
            stats[input_id] = 'TP'
            TP_count += 1
          else: # ground truth is negative or something else
            stats[input_id] = 'FP'
            FP_count += 1 
        elif args.negative_concept in consensus[input_id]: # annotation is negative
          if ground_truth[input_id] == [args.negative_concept]: # ground truth is negative
            stats[input_id] = 'TN'
            TN_count += 1
          elif args.positive_gt_label in ground_truth[input_id]: # ground truth is positive
            stats[input_id] = 'FN'
            FN_count += 1
        else: # annotaion is something else
          if (args.positive_gt_label in ground_truth[input_id]): # ground truth is positive
            stats[input_id] = 'FN'
            FN_count += 1

  logger.info("Stats computed.")
  return stats, TP_count, TN_count, FP_count, FN_count


def compute_non_positives(args, input_ids, consensus, ground_truth):
  ''' Compute how many times 3 or more labellers had not flaged any of positive labels '''

  # Count number of non positives from consensus 
  non_positive_count = 0
  for input_id in input_ids:
    if input_id in consensus:
      if consensus[input_id] is None:
        non_positive_count += 1
      elif not args.positive_concept_key in consensus[input_id]:
        non_positive_count += 1

  # Count number of non positives in ground_truth
  non_positive_count_gt = 0
  for input_id in input_ids:
    if not args.positive_gt_label in ground_truth[input_id]:
      non_positive_count_gt += 1 

  logger.info("Non positives computed.")
  return non_positive_count, non_positive_count_gt


def plot_results(args, input_count, no_gt_count, 
                 not_annotated_count, no_consensus_count,
                 TP_count, TN_count, FP_count, FN_count,
                 positive_gt_count, negative_gt_count, 
                 non_positive_count, non_positive_gt_count):
    ''' Print basic statistics in the console '''
    
    print("\n--------- Results ---------")
    print("Inputs --- total retrieved: {} | total kept: {}".
          format(input_count, input_count-no_gt_count))
    print("Groud truth --- not available {} | {}: {} | {}: {} | not {}: {}".
          format(no_gt_count,
                 args.negative_concept[2:], negative_gt_count,
                 args.positive_concept_key[2:], positive_gt_count,
                 args.positive_concept_key[2:], non_positive_gt_count))
    print("Labels --- not annotated: {} | no consensus: {} | not {}: {}".
          format(not_annotated_count, no_consensus_count, args.positive_concept_key[2:], non_positive_count))
    print("(TP) {} match: {}".format(args.positive_concept_key[2:], TP_count))
    print("(TN) {} match: {}".format(args.negative_concept[2:], TN_count))
    print("(FP) not {} but labeled as {}: {}".format(args.positive_concept_key[2:], args.positive_concept_key[2:], FP_count))
    print("(FN) {} but not labeled as {}: {}".format(args.positive_concept_key[2:], args.positive_concept_key[2:], FN_count))
    print("\n")

def get_misannotated_data(input_ids, ground_truth, annotations_meta, consensus, stats):
  ''' Get information about inputs that were mislabelled '''

  misannotated_ids = {}
  for input_id in input_ids:
    if input_id in stats:
      if stats[input_id] == 'FP' or stats[input_id] == 'FN':
        # Get initial information minus some fields
        input = input_ids[input_id]
        [input.pop(key) for key in ['results', 'source-file-line', 'group']]

        # Add information about results
        input['marker'] = stats[input_id]
        input['ground_truth'] = ground_truth[input_id]
        input['consensus'] = consensus[input_id]

        # Count different concepts
        concepts = [annotation['concept'] for annotation in annotations_meta[input_id]]
        input['annotations'] = {concept:concepts.count(concept) for concept in concepts}
        misannotated_ids[input_id] = input

        # Add raw information about annotations
        input['annotation_meta'] = annotations_meta[input_id]
  
  logger.info("Misannotated data extracted and stored.")

  return misannotated_ids
        

def save_input_metadata(args, input_ids):
  ''' Dump metadata about inputs (including ground truth) to a json file '''

  if args.save_input_meta:
    with open("{}/{}_{}_input_metadata.json".format(args.out_path, 
                                              args.app_name, 
                                              args.experiment_name.replace(' ', '-')
                                              ), 'w') as f:
      json.dump(input_ids, f)

def save_misannotated_data(args, misannotated_ids):
  ''' Dump data about misannotated inputa to a json file '''

  if args.save_misannotations:
    with open("{}/{}_{}_misannotations.json".format(args.out_path, 
                                              args.app_name, 
                                              args.experiment_name.replace(' ', '-')
                                              ), 'w') as f:
      json.dump(misannotated_ids, f)


def main(args, metadata):

  logger.info("----- Experiment {} - {} running -----".format(args.app_name, args.experiment_name))

  # Get input ids
  input_ids, input_count = get_input_ids(args, metadata)
  # Save metadata to re-use later if needed
  save_input_metadata(args, input_ids)

  # Get ground truth labels for every input and eliminate those that do not have it
  ground_truth, positive_gt_count, negative_gt_count, no_gt_count = get_ground_truth(args, input_ids)
  input_ids = remove_inputs_without_gt(input_ids, ground_truth)

  # Get annotations for every id together with their aggregations
  annotations, annotations_meta = get_annotations(args, metadata, input_ids)
  aggregated_annotations, not_annotated_count = aggregate_annotations(args, input_ids, annotations)

  # Compute consensus
  consensus, no_consensus_count = compute_consensus(args, input_ids, aggregated_annotations)

  # Compute matches with ground truth
  stats, TP_count, TN_count, FP_count, FN_count = compute_stats(args, input_ids, consensus, ground_truth) 

  # Count non positives
  non_positive_count, non_positive_count_gt = compute_non_positives(args, input_ids, consensus, ground_truth)

  # Plot statistics using computed values
  plot_results(args, input_count, no_gt_count,
               not_annotated_count, no_consensus_count, 
               TP_count, TN_count, FP_count, FN_count,
               positive_gt_count, negative_gt_count, 
               non_positive_count, non_positive_count_gt)

  # Get and save fails
  misannotated_ids = get_misannotated_data(input_ids, ground_truth, annotations_meta, consensus, stats)
  save_misannotated_data(args, misannotated_ids)


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
  parser.add_argument('--label_2_only',
                      default=True,
                      type=lambda x: (str(x).lower() == 'true'),
                      help="Preserve only ceoncepts starting with 2-")
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
  parser.add_argument('--save_misannotations',
                      default=True,
                      type=lambda x: (str(x).lower() == 'true'),
                      help="Save information about misannotated inputs in file or not.")

  args = parser.parse_args()

  metadata = (('authorization', 'Key {}'.format(args.api_key)),)

  if args.experiment == 1:
    # Experiment 1
    args.positive_concept_key = '2-HB'
    args.negative_concept = '2-not-hate'
    args.positive_gt_label = 'hate_speech'
    args.experiment_name = 'Hate Speech'

  elif args.experiment == 2:
    # Experiment 2
    args.positive_concept_key = '2-AD'
    args.negative_concept = '2-none-of-the-above'
    args.positive_gt_label = 'adult_&_explicit_sexual_content'
    args.experiment_name = 'AD'

  elif args.experiment == 3:
    # Experiment 3
    args.positive_concept_key = '2-OP'
    args.negative_concept = '2-none-of-the-above'
    args.positive_gt_label = 'obscenity_&_profanity'
    args.experiment_name = 'OP'

  elif args.experiment == 4:
    # Experiment 4
    args.positive_concept_key = '2-ID'
    args.negative_concept = '2-none-of-the-above'
    args.positive_gt_label = 'illegal_drugs/tobacco/e-cigarettes/vaping/alcohol'
    args.experiment_name = 'ID'

  main(args, metadata)