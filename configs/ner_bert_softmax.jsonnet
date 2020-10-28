// Configuration for a named entity recognization model based on:
//   Peters, Matthew E. et al. “Deep contextualized word representations.” NAACL-HLT (2018).
//   taken from https://gist.github.com/joelgrus/7cdb8fb2d81483a8d9ca121d9c617514 (slightly modified)

{
  local data_dir = "low-resource/src/data/distantly_labeled/",
  local bert_cased_archive = "bert-base-cased.tar.gz",

  local requires_grad = std.extVar("BERT_FINETUNE"),
  // learning rate of overall model.
  local LEARNING_RATE = std.extVar("LEARNING_RATE"),
  // dropout applied after pooling
  local DROPOUT = std.extVar("DROPOUT"),
  local CUDA = std.extVar("CUDA"),

  "dataset_reader": {
    "type": "jsonl_bert_reader",
    "tag_label": "ner",
    "coding_scheme": "IOB1",
    "token_indexers": {
      "tokens": {
         "type": "bert-pretrained",
          "pretrained_model": "bert-base-cased",
          "do_lowercase": false,
          "use_starting_offsets": true
      }
    }
  },

  "train_data_path": data_dir+ "amazon_train_basic.jsonl",
  "validation_data_path": data_dir+ "amazon_dev.jsonl",
  //"test_data_path": data_dir+ "amazon_test.jsonl",
  "model": {
    "type": "softmax_tagger_bert",
    "label_encoding": "BIO",
    "constrain_crf_decoding": true,
    "calculate_span_f1": false,
    "dropout": DROPOUT,
    "include_start_end_transitions": false,
    "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
          "tokens": ["tokens", "tokens-offsets"]
        },
        "token_embedders": {
            "tokens": {
                "type": "bert-pretrained",
                "pretrained_model": bert_cased_archive,
                //"top_layer_only": true,
                //"requires_grad": true,
            }
        }
    },
    "encoder": {
        "type": "lstm",
        "input_size": 768,
        "hidden_size": 200,
        "num_layers": 2,
        "dropout": DROPOUT,
        "bidirectional": true
    }
  },
  "iterator": {
    "type": "basic",
    "batch_size": 64
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": LEARNING_RATE //0.001
    },
    "validation_metric": "+f1-measure-overall",
    "num_serialized_models_to_keep": 3,
    "num_epochs": 5,
    "grad_norm": 5.0,
    "patience": 25,
    "cuda_device": [0,1]
  }
}