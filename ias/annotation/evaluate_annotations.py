import os
import json
import argparse
import itertools

# Import in the Clarifai gRPC based objects needed
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from google.protobuf.json_format import MessageToDict

# Construct the communications channel and the object stub to call requests on.
channel = ClarifaiChannel.get_json_channel()
stub = service_pb2_grpc.V2Stub(channel)


def get_input_ids(args, metadata):
  ''' Get list of all inputs (ids of videos that were uploaded) from the app '''

  # Get inputs
  list_inputs_response = stub.ListInputs(
                         service_pb2.ListInputsRequest(page=1, per_page=1000),
                         metadata=metadata
  )

  # Extract input ids
  input_ids = {}
  for input_object in list_inputs_response.inputs:
    json_obj = MessageToDict(input_object)
    input_ids[json_obj['id']] = json_obj['data']['metadata']
  print("Input ids fetched. Number of fetched inputs {}".format(len(input_ids)))

  # # ------ DEBUG CODE
  # input_ids_ = {}
  # for id in list(input_ids.keys())[0:50]:
  #   input_ids_[id] = input_ids[id]
  # input_ids = input_ids_
  # print("Number of selected inputs: ".format(len(input_ids)))
  # # ------ DEBUG CODE

  return input_ids


def get_ground_truth(args, input_ids):
  ''' Get list of ground truth concepts for every input id'''

  ground_truth = {}

  for input_id, input_val in input_ids.items():
    gt_keys = []
    # Extract all positive labels from results fields
    for gt_key, gt_val in input_val['results'].items():
      if gt_val == True:
        gt_keys.append(gt_key)
    ground_truth[input_id] = gt_keys

  # Count the number of positive and negative labels
  positive_count = sum([1 for input_id in input_ids if args.positive_gt_label in ground_truth[input_id]])
  negative_count = sum([1 for input_id in input_ids if args.positive_gt_label not in ground_truth[input_id]])

  print("Ground truth extracted.")

  # # ------ DEBUG CODE
  # # Compute list of unique labels
  # labels = list(itertools.chain(*[ground_truth[input_id] for input_id in input_ids]))
  # labels_count = {l:labels.count(l) for l in labels}
  # print("Ground truth labels are: ")
  # [print("\t{}: {}".format(k, v)) for k, v in labels_count.items()]
  # # ------ DEBUG CODE

  return ground_truth, positive_count, negative_count


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
                                per_page=20
                                ),
      metadata=metadata
    )
    # TODO: make requests in batches

    # Store annotations
    annotation = []
    annotation_meta = []
    for annotation_object in list_annotations_response.annotations:
      ao = MessageToDict(annotation_object)
      if 'concepts' in ao['data'] and len(ao['data']['concepts'])>0:
        # Store actual concept
        concept = ao['data']['concepts'][0]['name'] # TODO: check if id always = name
        if not args.label_2_only:
          annotation.append(concept)
        elif '2-' in concept:
            annotation.append(concept)
        # Store metadata
        meta = {'concept': concept, 'userId': ao['userId'], 'taskId': ao['taskId']}
        annotation_meta.append(meta)
        
    annotations[input_id] = annotation
    annotations_meta[input_id] = annotation_meta

    # Update max count variable
    annotation_nb_max = max(annotation_nb_max, len(list_annotations_response.annotations))

  print("Annotations fetched. Maximum number of annotation entries per input: {}".format(annotation_nb_max))

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

  # TODO: store aggregated annotations
  return aggregated_annotations, not_annotated_count


def compute_consensus(args, input_ids, aggregated_annotations):
  ''' Compute consensus among annotations for every input id based on aggregation'''

  # Variable to count how many times no full consensus has been reached
  no_consensus_count = 0

  def consesuns(value):
    return True if value >= args.consensus_count else False

  consensus = {}
  for input_id in input_ids:
      if input_id in aggregated_annotations:
        # Compute consensus
        aa = aggregated_annotations[input_id]
        consensus_exists = {k:consesuns(v) for k, v in aa.items()}

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
        
        # If consenus for any positive concept exists, chose the first positive concept
        # If not, chose negative concept
        if positive_concepts_concensus:
          consensus[input_id] = positive_concepts_concensus[0]
        else:
          consensus[input_id] = args.negative_concept

  return consensus, no_consensus_count


def compute_stats(args, input_ids, consensus, ground_truth):
  ''' Compute how many annotations match ground truth '''
  
  TP_count, TN_count, FP_count, FN_count = 0, 0, 0, 0

  stats = {}
  for input_id in input_ids:
    # Check first if this input was annotated (consensus varible contains only anotated inputs)
    if input_id in consensus:
      if consensus[input_id] is not None:
        if args.positive_concept_key in consensus[input_id]: # annotation is positive
        # if any(key in concept for key in args.positive_concept_keys): # annotation is positive
          if args.positive_gt_label in ground_truth[input_id]: # ground truth is positive
            stats[input_id] = 'TP'
            TP_count += 1
          else: # ground truth is negative
            stats[input_id] = 'FP'
            FP_count += 1 
        elif consensus[input_id] == args.negative_concept: # annotation is negative
          if args.positive_gt_label in ground_truth[input_id]: # ground truth is positive
            stats[input_id] = 'FN'
            FN_count += 1
          else: # ground truth is negative
            stats[input_id] = 'TN'
            TN_count += 1

  return stats, TP_count, TN_count, FP_count, FN_count


def plot_statistics(args, not_annotated_count, no_full_consensus_count, 
                    TP_count, TN_count, FP_count, FN_count,
                    positive_count, negative_count):
    ''' Print basic statistics in the console '''
               
    print(" Not Annotated: {}".format(not_annotated_count))
    print(" No Consensus: {}".format(no_full_consensus_count))
    print(" (TP) {} match: {}/{}".format(args.experiment_name, TP_count, positive_count))
    print(" (TN) Not {} match: {}/{}".format(args.experiment_name, TN_count, negative_count))
    print(" (FP) Not {} but labeled as {}: {}/{}".format(args.experiment_name, args.experiment_name, FP_count, negative_count))
    print(" (FN) {} but not labeled as {}: {}/{}".format(args.experiment_name, args.experiment_name, FN_count, positive_count))
    print(" ")

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
        input['ground_truth'] = ground_truth[input_id][0]
        input['consensus'] = consensus[input_id]

        # Count different concepts
        concepts = [annotation['concept'] for annotation in annotations_meta[input_id]]
        input['annotations'] = {concept:concepts.count(concept) for concept in concepts}
        misannotated_ids[input_id] = input

        # Add raw information about annotations
        input['annotation_meta'] = annotations_meta[input_id]
  
  return misannotated_ids
        

def save_input_metadata(args, input_ids):
  ''' Dump metadata about inputs (including ground truth) to a json file '''

  if args.save_input_meta:
    with open("{}/{}_{}_input_metadata.json".format(args.out_path, 
                                              args.app_name.lower(), 
                                              args.experiment_name.replace(' ', '-')
                                              ), 'w') as f:
      json.dump(input_ids, f)

def save_misannotated_data(args, misannotated_ids):
  ''' Dump data about misannotated inputa to a json file '''

  if args.save_misannotations:
    with open("{}/{}_{}_misannotations.json".format(args.out_path, 
                                              args.app_name.lower(), 
                                              args.experiment_name.replace(' ', '-')
                                              ), 'w') as f:
      json.dump(misannotated_ids, f)


def main(args, metadata):

  print("------- Experiment {}-{} running -------".format(args.app_name, args.experiment_name))

  # Get input ids
  input_ids = get_input_ids(args, metadata)
  # Save metadata to re-use later if needed
  save_input_metadata(args, input_ids)

  # Get ground truth labels for every id
  ground_truth, positive_count, negative_count = get_ground_truth(args, input_ids)

  # Get annotations for every id together with their aggregations
  annotations, annotations_meta = get_annotations(args, metadata, input_ids)
  aggregated_annotations, not_annotated_count = aggregate_annotations(args, input_ids, annotations)

  # Compute consensus
  consensus, no_full_consensus_count = compute_consensus(args, input_ids, aggregated_annotations)

  # Compute matches with ground truth
  stats, TP_count, TN_count, FP_count, FN_count = compute_stats(args, input_ids, consensus, ground_truth) 

  # Plot statistics using computed values
  print("------- Results -------")
  plot_statistics(args, not_annotated_count, no_full_consensus_count, 
                  TP_count, TN_count, FP_count, FN_count,
                  positive_count, negative_count)

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
                      default=2, 
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
  parser.add_argument('--out_path', 
                      default="ias/annotation/output", 
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