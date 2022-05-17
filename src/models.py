# 幹：讓 unit 從 0 ~ 499
# 幹：增加 model args

from typing import Optional
import torch, torch.nn as nn

from fairseq.models.transformer import TransformerEncoder
from fairseq.models.transformer import TransformerModel
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from .tasks import load_lengthaug_langpair_dataset
from .torch_cif import cif_function
from .utils import mask_generator
from fairseq.models.transformer import base_architecture

class BottleneckedTransformerEncoderPrototype(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        embed_dim = embed_tokens.embedding_dim
        # 幹：word_extractor 拉出來
        # 幹：dictionary 換？
        word_extractor = cif_function
        self.post_initialization(embed_dim, word_extractor)
        self.use_self = None  # 幹！拿掉 TODO

    def post_initialization(self, embed_dim, word_extractor):
        # == encoder == #
        self.alpha_predictor = nn.Linear(embed_dim, 1)  # TODO: check!
        self.word_extractor = word_extractor
        self.length_predictor = nn.Linear(embed_dim, 1)

    def forward(
        self,
        src_tokens,
        word_length_tensor: Optional[torch.Tensor] = None,  # <--
        alpha_values: Optional[torch.Tensor] = None,  # <--
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            src_tokens, 
            word_length_tensor, alpha_values,  # <--
            src_lengths, return_all_hiddens, token_embeddings
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        word_length_tensor: Optional[torch.Tensor] = None,  # <--
        alpha_values: Optional[torch.Tensor] = None,  # <--
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError(
            "Should be overloaded!"
        )

    def sent_retriever(self, 
        encoder__last_hidden_state, 
        word_length_tensor=None,
        alpha_values=None,
        padding_mask=None,
        return_all=False,
        return_original=False,
      ):
        if alpha_values is None:
            alpha_values = self.alpha_predictor(encoder__last_hidden_state)
            alpha_values = alpha_values.squeeze(-1).sigmoid()  # B x S

        if word_length_tensor is None or self.use_self:
            # print("No given! self predict")
            word_length_tensor = alpha_values.sum(-1).long()
        else:
            # print("Wordlen given")
            # predicted_word_length_tensor = alpha_values.sum(-1).long()
            pass

        encoder__word_representations_CIF = (
            self.word_extractor(
                encoder__last_hidden_state,
                alpha=alpha_values,
                padding_mask=padding_mask,
                target_lengths=word_length_tensor,
            )
        )
        # TODO: Keep all CIF
        [encoder_word_representation] = encoder__word_representations_CIF['cif_out']
        [pred_word_lengths] = encoder__word_representations_CIF['alpha_sum']
        encoder_word_representation = encoder_word_representation.contiguous()
        # pred_word_lengths = pred_word_lengths.contiguous()
        # length_loss = 0.
        length_loss = ((pred_word_lengths, word_length_tensor,)
            if word_length_tensor is not None else
            (pred_word_lengths, pred_word_lengths,))
            # aliased as `encoder_word_representation`
            # FIXME: distributed problem!
            # TODO: add other CIF ouptuts!
        # length_loss = length_loss.contiguous()

        encoder_output_attention_mask = (
            # mask_generator(word_length_tensor) 
            mask_generator(word_length_tensor) 
            if word_length_tensor is not None else
            mask_generator(pred_word_lengths))
            # TODO: check length prediction!
            
        return (
            encoder_word_representation, 
            encoder_output_attention_mask, 
            length_loss,
            pred_word_lengths,
            encoder__word_representations_CIF if return_all else None,
            # encoder__last_hidden_state if return_original else None,
            encoder__last_hidden_state if return_original else None,
        )


class BottleneckedTransformerEncoder(
      BottleneckedTransformerEncoderPrototype
  ):
    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        word_length_tensor: Optional[torch.Tensor] = None,  # <--
        alpha_values: Optional[torch.Tensor] = None,  # <--
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        if not getattr(self.args, "use_self", False):
            word_length_tensor = None

        if not getattr(self.args, "skip_bottleneck", False):
            # T x B x C -> B x T x C
            x = x.transpose(0, 1)
            (
                x,  # B x T x C'
                out_attention_mask,
                length_loss,
                pred_word_lengths,
                encoder__word_representations_CIF,
                encoder__last_hidden_state,
            ) = self.sent_retriever(
                encoder__last_hidden_state=x,  # B x T x C
                word_length_tensor=word_length_tensor,
                alpha_values=alpha_values,
                padding_mask=encoder_padding_mask,
                # return_all=False,
                # return_original=False,
            )

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)
            encoder_padding_mask = (1 - out_attention_mask).bool()

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
                      .sum(dim=1, dtype=torch.int32)
                      .reshape(-1, 1)
                      .contiguous())
                      

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

class FrontBottleneckedTransformerEncoder(
    BottleneckedTransformerEncoderPrototype
):
    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        word_length_tensor: Optional[torch.Tensor] = None,  # <--
        alpha_values: Optional[torch.Tensor] = None,  # <--
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
      ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        # print('\n''\033[01;31m''=== ¡breakpoint! ===''\033[0m''\n'); breakpoint()  # from IPython import embed; embed()
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # x: B x T x C
        # $$$$$$$$$$$$$$$$$$$$$$$$$$
        (
            x,  # B x T x C'
            out_attention_mask,
            length_loss,
            pred_word_lengths,
            encoder__word_representations_CIF,
            encoder__last_hidden_state,
        ) = self.sent_retriever(
            encoder__last_hidden_state=x,  # B x T x C
            word_length_tensor=word_length_tensor,
            alpha_values=alpha_values,
            padding_mask=encoder_padding_mask,
            # return_all=False,
            # return_original=False,
        )
        encoder_padding_mask = (1 - out_attention_mask).bool()
        encoder_embedding = x

        # B x T x C' -> T x B x C'
        x = x.transpose(0, 1)
        
        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
                      .sum(dim=1, dtype=torch.int32)
                      .reshape(-1, 1)
                      .contiguous())
                      
        # B x T x C' -> T x B x C'
        x = x.transpose(0, 1)

        return {
            "encoder_out": [x],  # T x B x C'
            "encoder_padding_mask": [(1 - out_attention_mask).bool()],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C'
            "encoder_states": encoder_states,  # List[T x B x C']
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

@register_model("wordlen_transformer")
class BottleneckedTransformerModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--fix-encoder', action='store_true')
        parser.add_argument('--use-self', action='store_true')
        parser.add_argument('--skip-bottleneck', action='store_true')
        super(BottleneckedTransformerModel, 
              BottleneckedTransformerModel).add_args(parser)
        
    
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return BottleneckedTransformerEncoder(args, src_dict, embed_tokens)
    
    @classmethod
    def build_model(cls, args, task):
        model = super().build_model(args, task)
        if getattr(args, "fix_encoder", False):
            args.fix_encoder = True
            model.fix_encoder_()
        return model
        
    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        word_length_tensor,  # <--
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, 
            src_lengths=src_lengths, 
            word_length_tensor=word_length_tensor,  # <--
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    def fix_encoder_(self, to_fix=True):
        self.encoder.requires_grad_(not to_fix)

@register_model("before_wordlen_transformer")
class FrontBottleneckedTransformerModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--fix-encoder', action='store_true')
        parser.add_argument('--use-self', action='store_true')
        parser.add_argument('--skip-bottleneck', action='store_true')
        super(FrontBottleneckedTransformerModel, 
              FrontBottleneckedTransformerModel).add_args(parser)
    
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return FrontBottleneckedTransformerEncoder(args, src_dict, embed_tokens)
        
    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        word_length_tensor,  # <--
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
      ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, 
            src_lengths=src_lengths, 
            word_length_tensor=word_length_tensor,  # <--
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    def fix_encoder_(self, to_fix=True):
        self.encoder.requires_grad_(not to_fix)

@register_model_architecture(
    model_name="wordlen_transformer", 
    arch_name="iwslt_wordlen_transformer")
def iwslt_wordlen_transformer(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dbim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)

