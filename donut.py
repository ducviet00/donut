import torch
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel

from transformers import AutoConfig, SwinModel, AutoModelForCausalLM

import math
import torch.nn.functional as F


class DonutModel(VisionEncoderDecoderModel):
    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        base_config: VisionEncoderDecoderConfig = None,
        *model_args,
        **kwargs,
    ) -> VisionEncoderDecoderModel:
        encoder_config = AutoConfig.from_pretrained(
            encoder_pretrained_model_name_or_path
        )
        encoder_config.architectures = ['SwinModel']
        encoder_config.depths = base_config.encoder.depths
        encoder_config.image_size = base_config.encoder.image_size
        encoder_config.window_size = base_config.encoder.window_size
        encoder_config.num_labels = 0
        encoder = SwinModel(encoder_config)

        encoder_state_dict = SwinModel.from_pretrained(
            encoder_pretrained_model_name_or_path
        ).state_dict()
        new_encoder_state_dict = encoder.state_dict()
        for x in new_encoder_state_dict:
            if x.endswith("relative_position_index"):
                pass
            elif x.endswith("relative_position_bias_table"):
                pos_bias = encoder_state_dict[x].unsqueeze(0)[0]
                old_len = int(math.sqrt(len(pos_bias)))
                new_len = int(2 * base_config.encoder.window_size - 1)
                pos_bias = pos_bias.reshape(1, old_len, old_len, -1).permute(0, 3, 1, 2)
                pos_bias = F.interpolate(
                    pos_bias,
                    size=(new_len, new_len),
                    mode="bicubic",
                    align_corners=False,
                )
                new_encoder_state_dict[x] = (
                    pos_bias.permute(0, 2, 3, 1).reshape(1, new_len**2, -1).squeeze(0)
                )
            else:
                new_encoder_state_dict[x] = encoder_state_dict[x]

        encoder.load_state_dict(new_encoder_state_dict)

        decoder_config = AutoConfig.from_pretrained(
            decoder_pretrained_model_name_or_path
        )
        decoder_config.is_encoder_decoder = base_config.decoder.is_encoder_decoder
        decoder_config.is_decoder = base_config.decoder.is_decoder
        decoder_config.add_cross_attention = base_config.decoder.add_cross_attention
        decoder_config.decoder_layers = base_config.decoder.decoder_layers
        decoder_config.max_position_embeddings = (
            base_config.decoder.max_position_embeddings
        )
        decoder_config.vocab_size = base_config.decoder.vocab_size
        decoder_config.scale_embedding = base_config.decoder.scale_embedding
        decoder_config.add_final_layer_norm = base_config.decoder.add_final_layer_norm
        decoder_config.max_length = base_config.decoder.max_length

        decoder = AutoModelForCausalLM.from_config(decoder_config)

        bart_state_dict = AutoModelForCausalLM.from_pretrained(
            decoder_pretrained_model_name_or_path
        ).state_dict()
        new_bart_state_dict = decoder.state_dict()
        for x in new_bart_state_dict:
            if (
                x.endswith("embed_positions.weight")
                and base_config.decoder.max_position_embeddings != 1024
            ):
                new_bart_state_dict[x] = torch.nn.Parameter(
                    cls.resize_bart_abs_pos_emb(
                        bart_state_dict[x],
                        base_config.decoder.max_position_embeddings
                        + 2,  # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                    )
                )
            elif x.endswith("embed_tokens.weight") or x.endswith("lm_head.weight"):
                new_bart_state_dict[x] = bart_state_dict[x][
                    : base_config.decoder.vocab_size, :
                ]
            else:
                new_bart_state_dict[x] = bart_state_dict[x]
        decoder.load_state_dict(new_bart_state_dict)

        # instantiate config with corresponding kwargs
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
            encoder.config, decoder.config, **kwargs
        )

        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        return cls(encoder=encoder, decoder=decoder, config=config)

    @staticmethod
    def resize_bart_abs_pos_emb(weight: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Resize position embeddings
        Truncate if sequence length of Bart backbone is greater than given max_length,
        else interpolate to max_length
        """
        if weight.shape[0] > max_length:
            weight = weight[:max_length, ...]
        else:
            weight = (
                F.interpolate(
                    weight.permute(1, 0).unsqueeze(0),
                    size=max_length,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .permute(1, 0)
            )
        return weight
