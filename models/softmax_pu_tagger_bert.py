from typing import Dict, Optional, List, Any

from overrides import overrides
import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure

from low_resource.metrics import F1Measure


@Model.register("softmax_pu_tagger_bert")
class SoftmaxPUTaggerBert(Model):
    """
    The ``CrfTagger`` encodes a sequence of text with a ``Seq2SeqEncoder``,
    then uses a Conditional Random Field model to predict a tag for each token in the sequence.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the tokens ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder that we will use in between embedding tokens and predicting output tags.
    label_namespace : ``str``, optional (default=``labels``)
        This is needed to compute the SpanBasedF1Measure metric.
        Unless you did something unusual, the default value should be what you want.
    feedforward : ``FeedForward``, optional, (default = None).
        An optional feedforward layer to apply after the encoder.
    label_encoding : ``str``, optional (default=``None``)
        Label encoding to use when calculating span f1 and constraining
        the CRF at decoding time . Valid options are "BIO", "BIOUL", "IOB1", "BMES".
        Required if ``calculate_span_f1`` or ``constrain_crf_decoding`` is true.
    include_start_end_transitions : ``bool``, optional (default=``True``)
        Whether to include start and end transition parameters in the CRF.
    constrain_crf_decoding : ``bool``, optional (default=``None``)
        If ``True``, the CRF is constrained at decoding time to
        produce valid sequences of tags. If this is ``True``, then
        ``label_encoding`` is required. If ``None`` and
        label_encoding is specified, this is set to ``True``.
        If ``None`` and label_encoding is not specified, it defaults
        to ``False``.
    calculate_span_f1 : ``bool``, optional (default=``None``)
        Calculate span-level F1 metrics during training. If this is ``True``, then
        ``label_encoding`` is required. If ``None`` and
        label_encoding is specified, this is set to ``True``.
        If ``None`` and label_encoding is not specified, it defaults
        to ``False``.
    dropout:  ``float``, optional (default=``None``)
    verbose_metrics : ``bool``, optional (default = False)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 label_namespace: str = "labels",
                 feedforward: Optional[FeedForward] = None,
                 label_encoding: Optional[str] = None,
                 calculate_span_f1: bool = None,
                 dropout: Optional[float] = None,
                 verbose_metrics: bool = False,
                 prior: Optional[float] = 0.01,
                 gamma: Optional[float] = 1,
                 m: Optional[float] = 1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size(label_namespace)
        self.encoder = encoder
        self._verbose_metrics = verbose_metrics
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self._feedforward = feedforward

        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
        else:
            output_dim = self.encoder.get_output_dim()
        self.tag_projection_layer = TimeDistributed(Linear(output_dim,
                                                           self.num_tags))

        self.entroyLoss = nn.CrossEntropyLoss()

        if calculate_span_f1 is None:
            calculate_span_f1 = label_encoding is not None

        self.label_encoding = label_encoding


        self.prior = prior #0.012
        self.beta = 0.0
        self.gamma = gamma #1
        self.m = m  #1


        self.metrics = {
            "accuracy": CategoricalAccuracy()
        }
        self.calculate_span_f1 = calculate_span_f1
        if calculate_span_f1:
            if not label_encoding:
                raise ConfigurationError("calculate_span_f1 is True, but "
                                         "no label_encoding was specified.")
            self._f1_metric = SpanBasedF1Measure(vocab,
                                                 tag_namespace=label_namespace,
                                                 label_encoding=label_encoding)
        else:
            self._f1_metric = F1Measure(vocabulary=self.vocab)

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        if feedforward is not None:
            check_dimensions_match(encoder.get_output_dim(), feedforward.get_input_dim(),
                                   "encoder output dim", "feedforward input dim")
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:


        embedded_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)

        encoded_text = self.encoder(embedded_text_input, mask)

        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        if self._feedforward is not None:
            encoded_text = self._feedforward(encoded_text)

        logits = self.tag_projection_layer(encoded_text)

        predicted_tags = torch.argmax(logits, dim=-1)

        output = {"logits": logits, "mask": mask, "label_tags": tags,
                  "tags": predicted_tags}  # , "num_tags": 2self.num_tags

        device = logits.get_device()
        logits_dim = logits.size()[-1]
        if tags is not None:
            # Add negative log-likelihood as loss

            active_logits_mask = (mask == 1).unsqueeze(-1).expand(*logits.size())

            logits_All = logits.masked_select(active_logits_mask).view(-1, logits_dim)
            tags = tags[:, :mask.size()[-1]]
            tags_all = tags.masked_select(mask == 1)

            ##this is PU algorithm
            logits_P = logits_All.masked_select((tags_all > 0).unsqueeze(-1).expand(*logits_All.size()))
            logits_U = logits_All.masked_select((tags_all == 0).unsqueeze(-1).expand(*logits_All.size())).view(-1,
                                                                                                                   logits_dim)
            if len(logits_P) > 0:
                logits_P = logits_P.view(-1, logits_dim)
                log_likelihood_p = self.entroyLoss(logits_P, torch.ones(logits_P.size()[0]).long().to(device))
                log_likelihood_p_n = self.entroyLoss(logits_P, torch.zeros(logits_P.size()[0]).long().to(device))
                pRisk = log_likelihood_p * (logits_P.size()[0])
                pNRisk = log_likelihood_p_n * (logits_P.size()[0])
            else:
                pRisk = torch.FloatTensor([0]).squeeze().to(device)
                pNRisk = torch.FloatTensor([0]).squeeze().to(device)

            if len(logits_U) == 0:
                uRisk = torch.FloatTensor([0]).squeeze().to(device)
            else:
                log_likelihood_u = self.entroyLoss(logits_U, torch.zeros(logits_U.size()[0]).long().to(device))
                uRisk = log_likelihood_u * (logits_U.size()[0])

            nRisk = uRisk - self.prior * pNRisk

            if nRisk < self.beta:
                risk = pRisk
            else:
                risk = self.m * pRisk + nRisk

            output["loss"] = risk

            # feed into the metrics
            class_probabilities = logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, tags, mask.float())
            # if self.calculate_span_f1:
            self._f1_metric(class_probabilities, tags, mask.float())
        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]
        return output

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """

        tags_list = output_dict["tags"].cpu().numpy().tolist()
        glod_tags_list = output_dict["label_tags"].cpu().numpy().tolist()

        predict_tags = [
            [self.vocab.get_token_from_index(tag, namespace=self.label_namespace)
             for tag in instance_tags]
            for instance_tags in tags_list
        ]
        output_dict["tags"] = predict_tags
        output_dict["label_tags"] = [
            [self.vocab.get_token_from_index(tag, namespace=self.label_namespace)
             for tag in instance_tags]
            for instance_tags in glod_tags_list
        ]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.metrics.items()}

        f1_dict = self._f1_metric.get_metric(reset=reset)
        if self._verbose_metrics:
            metrics_to_return.update(f1_dict)
        else:
            metrics_to_return.update({
                x: y for x, y in f1_dict.items() if
                "overall" in x})
        return metrics_to_return
