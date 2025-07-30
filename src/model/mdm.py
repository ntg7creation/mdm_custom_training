import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
from model.BERT.BERT_encoder import load_bert
from utils.misc import WeightedSum


class MDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()
        
        self.legacy = legacy                  # Enables legacy support for older models or behaviors (#!not used in T2M typically)
        self.modeltype = modeltype            # Identifier for the type of model (#!not used in T2M)
        self.njoints = njoints                # Number of joints in the input motion skeleton ✅ used in T2M
        self.nfeats = nfeats                  # Number of features per joint (e.g., 3 for XYZ, 6 for rot6d) ✅ used in T2M
        self.num_actions = num_actions        # Number of possible action labels (#!not used in T2M — used in Action-to-Motion)
        self.data_rep = data_rep              # Type of motion representation ('rot6d', 'hml_vec', etc.) ✅ used in T2M
        self.dataset = dataset                # Dataset name used to adjust processing logic ('humanml', 'kit', etc.) ✅ used in T2M


        self.pose_rep = pose_rep              # Representation of pose used internally (e.g., 'rot6d') ✅ used in T2M
        self.glob = glob                      # Whether to include global position (root joint) ✅ used in T2M
        self.glob_rot = glob_rot              # Whether to include global rotation of the body ✅ used in T2M
        self.translation = translation        # Whether to include translational motion of the root joint ✅ used in T2M


        self.latent_dim = latent_dim          # Dimensionality of embeddings and transformer layers ✅ used in T2M

        self.ff_size = ff_size                # Size of the feed-forward layer inside transformer blocks ✅ used in T2M
        self.num_layers = num_layers          # Number of layers in the transformer or GRU ✅ used in T2M
        self.num_heads = num_heads            # Number of attention heads in the transformer ✅ used in T2M
        self.dropout = dropout                # Dropout rate for transformer layers ✅ used in T2M


        self.ablation = ablation                      # Optional flag for ablation experiments (#!not used in T2M by default)
        self.activation = activation                  # Activation function used in the transformer (e.g., 'gelu') ✅ used in T2M
        self.clip_dim = clip_dim                      # Output dimension of the CLIP/BERT encoder (e.g., 512 or 768) ✅ used in T2M
        self.action_emb = kargs.get('action_emb', None)  # Optional setting for action embedding (#!not used in T2M — used in A2M)
        self.input_feats = self.njoints * self.nfeats # Flattened input feature size per frame ✅ used in T2M

        self.normalize_output = kargs.get('normalize_encoder_output', False)  # Option to normalize transformer output (#!not typically used in T2M)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')            # Type of conditioning ('text', 'action', or 'no_cond') ✅ used in T2M
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)         # Probability of masking condition (for classifier-free guidance) ✅ used in T2M
        self.mask_frames = kargs.get('mask_frames', False)            # Whether to mask invalid frames (#!optional, not always used in T2M)
        self.arch = arch                                               # Model architecture: 'trans_enc', 'trans_dec', or 'gru' ✅ used in T2M
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0  # Extra GRU embedding dimension (#!not used in T2M if using transformer)
        
        #hml_vec is  a custom 263-dimensional vector representation used specifically in the HumanML3D dataset
        # Initializes the module that prepares motion input for the transformer.
        # - self.data_rep: Specifies the type of motion representation (e.g., 'hml_vec', 'rot6d'),
        #                  which determines how the input should be interpreted and embedded.
        # - self.input_feats: The number of raw features per frame, typically njoints × nfeats = 263
        #                     in HumanML3D (i.e., 22 joints with multiple features).
        # - self.gru_emb_dim: Additional embedding space used only when GRU is selected
        #                     (0 in T2M since transformers are used).
        # - self.latent_dim: The transformer’s input embedding dimension (e.g., 512),
        #                    into which all motion frames are projected.
        #
        # The InputProcess module flattens each motion frame to [263] and projects it into [512]
        # so that it can be passed as a sequence into the transformer.

        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)  # Prepares motion input for transformer ✅ used in T2M

        self.emb_policy = kargs.get('emb_policy', 'add')              # Embedding fusion strategy: 'add' or 'concat' ✅ used in T2M

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=kargs.get('pos_embed_max_len', 5000))  # Positional encoding for temporal order ✅ used in T2M
        self.emb_trans_dec = emb_trans_dec                            # Whether to inject embedding as class token (used only in 'trans_dec') (#!only used for some T2M variants)

        self.pred_len = kargs.get('pred_len', 0)  # Number of frames to predict in prefix completion (✅ used in T2M prefix setting)
        self.context_len = kargs.get('context_len', 0)  # Number of context frames provided before prediction (✅ used in T2M prefix setting)
        self.total_len = self.pred_len + self.context_len  # Total length for prefix models (context + prediction)
        self.is_prefix_comp = self.total_len > 0  # Flag for whether this is a prefix-completion model (✅ used in T2M with prefix mode)
        self.all_goal_joint_names = kargs.get('all_goal_joint_names', [])  # Names of joints to be targeted during conditioning (#!not used in plain T2M)

        
        self.multi_target_cond = kargs.get('multi_target_cond', False)  # Enable multi-joint target conditioning (#!not used in standard T2M)
        self.multi_encoder_type = kargs.get('multi_encoder_type', 'multi')  # Type of target joint encoder (multi/single/split) (#!not used in T2M)
        self.target_enc_layers = kargs.get('target_enc_layers', 1)  # Number of layers in the target encoder (#!not used in T2M)

        # Initialize the appropriate joint target conditioning module if enabled
        if self.multi_target_cond:
            if self.multi_encoder_type == 'multi':
                self.embed_target_cond = EmbedTargetLocMulti(self.all_goal_joint_names, self.latent_dim)
            elif self.multi_encoder_type == 'single':
                self.embed_target_cond = EmbedTargetLocSingle(self.all_goal_joint_names, self.latent_dim, self.target_enc_layers)
            elif self.multi_encoder_type == 'split':
                self.embed_target_cond = EmbedTargetLocSplit(self.all_goal_joint_names, self.latent_dim, self.target_enc_layers)

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")  # Logs architecture being initialized (transformer encoder)
            seqTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,             # Transformer layer input/output dim (e.g., 512)
                nhead=self.num_heads,                # Number of attention heads
                dim_feedforward=self.ff_size,        # Feedforward network size inside each transformer block
                dropout=self.dropout,                # Dropout rate
                activation=self.activation           # Activation function, e.g., GELU
            )

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(
                d_model=self.latent_dim,          # Size of input/output embeddings (e.g., 512)
                nhead=self.num_heads,             # Number of self-attention heads
                dim_feedforward=self.ff_size,     # Size of inner feedforward layer (e.g., 1024)
                dropout=self.dropout,             # Dropout rate
                activation=activation             # Activation function, e.g., 'gelu'
            )
            self.seqTransDecoder = nn.TransformerDecoder(
                seqTransDecoderLayer,
                num_layers=self.num_layers        # Number of decoder layers in the stack
            )
        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
          raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        # Embeds the diffusion timestep (t) into a vector matching latent_dim (e.g., 512)
        # This embedding is added to other embeddings (like text or motion) so the model
        # knows how much noise is present in the current training/sampling step
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                # We support CLIP encoder and DistilBERT
                print('EMBED TEXT')
                
                self.text_encoder_type = kargs.get('text_encoder_type', 'clip')
                
                if self.text_encoder_type == "clip":
                    print('Loading CLIP...')
                    self.clip_version = clip_version
                    self.clip_model = self.load_and_freeze_clip(clip_version)
                    self.encode_text = self.clip_encode_text
                elif self.text_encoder_type == 'bert':
                    assert self.arch == 'trans_dec'
                    # assert self.emb_trans_dec == False # passing just the time embed so it's fine
                    print("Loading BERT...")
                    # bert_model_path = 'model/BERT/distilbert-base-uncased'
                    bert_model_path = 'distilbert/distilbert-base-uncased'
                    self.clip_model = load_bert(bert_model_path)  # Sorry for that, the naming is for backward compatibility
                    self.encode_text = self.bert_encode_text
                    self.clip_dim = 768
                else:
                    raise ValueError('We only support [CLIP, BERT] text encoders') 
                
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[-2]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(1, bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def clip_encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float().unsqueeze(0)
    
    def bert_encode_text(self, raw_text):
        # enc_text = self.clip_model(raw_text)
        # enc_text = enc_text.permute(1, 0, 2)
        # return enc_text
        enc_text, mask = self.clip_model(raw_text)  # self.clip_model.get_last_hidden_state(raw_text, return_mask=True)  # mask: False means no token there
        enc_text = enc_text.permute(1, 0, 2)
        mask = ~mask  # mask: True means no token there, we invert since the meaning of mask for transformer is inverted  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        return enc_text, mask

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        time_emb = self.embed_timestep(timesteps)  # [1, bs, d]

        if 'target_cond' in y.keys():
            # NOTE: We don't use CFG for joints - but we do wat to support uncond sampling for generation and eval!
            time_emb += self.mask_cond(self.embed_target_cond(y['target_cond'], y['target_joint_names'], y['is_heading'])[None], force_mask=y.get('target_uncond', False))  # For uncond support and CFG
            # time_emb += self.embed_target_cond(y['target_cond'], y['target_joint_names'], y['is_heading'])[None]  

        # Build input for prefix completion
        if self.is_prefix_comp:
            x = torch.cat([y['prefix'], x], dim=-1)
            y['mask'] = torch.cat([torch.ones([bs, 1, 1, self.context_len], dtype=y['mask'].dtype, device=y['mask'].device), 
                                   y['mask']], dim=-1)

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            if 'text_embed' in y.keys():  # caching option
                enc_text = y['text_embed']
            else:
                enc_text = self.encode_text(y['text'])
            if type(enc_text) == tuple:
                enc_text, text_mask = enc_text
                if text_mask.shape[0] == 1 and bs > 1:  # casting mask for the single-prompt-for-all case
                    text_mask = torch.repeat_interleave(text_mask, bs, dim=0)
            text_emb = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))  # casting mask for the single-prompt-for-all case
            if self.emb_policy == 'add':
                emb = text_emb + time_emb
            else:
                emb = torch.cat([time_emb, text_emb], dim=0)
                text_mask = torch.cat([torch.zeros_like(text_mask[:, 0:1]), text_mask], dim=1)
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb = time_emb + self.mask_cond(action_emb, force_mask=force_mask)
        if self.cond_mode == 'no_cond': 
            # unconstrained
            emb = time_emb

        if self.arch == 'gru':
            x_reshaped = x.reshape(bs, njoints*nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)     #[#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)      #[bs, d, #frames]
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  #[bs, d, 1, #frames]
            x = torch.cat((x_reshaped, emb_gru), axis=1)  #[bs, d+joints*feat, 1, #frames]

        x = self.input_process(x)

        # TODO - move to collate
        frames_mask = None
        is_valid_mask = y['mask'].shape[-1] > 1  # Don't use mask with the generate script
        if self.mask_frames and is_valid_mask:
            frames_mask = torch.logical_not(y['mask'][..., :x.shape[0]].squeeze(1).squeeze(1)).to(device=x.device)
            if self.emb_trans_dec or self.arch == 'trans_enc':
                step_mask = torch.zeros((bs, 1), dtype=torch.bool, device=x.device)
                frames_mask = torch.cat([step_mask, frames_mask], dim=1)

        if self.arch == 'trans_enc':
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq, src_key_padding_mask=frames_mask)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                xseq = torch.cat((time_emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]

            if self.text_encoder_type == 'clip':
                output = self.seqTransDecoder(tgt=xseq, memory=emb, tgt_key_padding_mask=frames_mask)
            elif self.text_encoder_type == 'bert':
                output = self.seqTransDecoder(tgt=xseq, memory=emb, memory_key_padding_mask=text_mask, tgt_key_padding_mask=frames_mask)  # Rotem's bug fix
            else:
                raise ValueError()

            if self.emb_trans_dec:
                output = output[1:] # [seqlen, bs, d]

        elif self.arch == 'gru':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)

        # Extract completed suffix
        if self.is_prefix_comp:
            output = output[self.context_len:]
            y['mask'] = y['mask'][..., self.context_len:]
        
        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output


    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output
    
class EmbedTargetLocSingle(nn.Module):
    def __init__(self, all_goal_joint_names, latent_dim, num_layers=1):
        super().__init__()
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']
        self.target_cond_dim = len(self.extended_goal_joint_names) * 4  # 4 => (x,y,z,is_valid)
        self.latent_dim = latent_dim
        _layers = [nn.Linear(self.target_cond_dim, self.latent_dim)]
        for _ in range(num_layers):
            _layers += [nn.SiLU(), nn.Linear(self.latent_dim, self.latent_dim)]
        self.mlp = nn.Sequential(*_layers)

    def forward(self, input, target_joint_names, target_heading):
        # TODO - generate validity from outside the model
        validity = torch.zeros_like(input)[..., :1]
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names
            for j in sample_joint_names_w_heading:
                validity[sample_idx, self.extended_goal_joint_names.index(j)] = 1.

        mlp_input = torch.cat([input, validity], dim=-1).view(input.shape[0], -1)
        return self.mlp(mlp_input)


class EmbedTargetLocSplit(nn.Module):
    def __init__(self, all_goal_joint_names, latent_dim, num_layers=1):
        super().__init__()
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']
        self.target_cond_dim = 4
        self.latent_dim = latent_dim
        self.splited_dim = self.latent_dim // len(self.extended_goal_joint_names)
        assert self.latent_dim % len(self.extended_goal_joint_names) == 0
        self.mini_mlps = nn.ModuleList()
        for _ in self.extended_goal_joint_names:
            _layers = [nn.Linear(self.target_cond_dim, self.splited_dim)]
            for _ in range(num_layers):
                _layers += [nn.SiLU(), nn.Linear(self.splited_dim, self.splited_dim)]
            self.mini_mlps.append(nn.Sequential(*_layers))

    def forward(self, input, target_joint_names, target_heading):
        # TODO - generate validity from outside the model
        validity = torch.zeros_like(input)[..., :1]
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names
            for j in sample_joint_names_w_heading:
                validity[sample_idx, self.extended_goal_joint_names.index(j)] = 1.

        mlp_input = torch.cat([input, validity], dim=-1)
        mlp_splits = [self.mini_mlps[i](mlp_input[:, i]) for i in range(mlp_input.shape[1])] 
        return torch.cat(mlp_splits, dim=-1)
  
class EmbedTargetLocMulti(nn.Module):
    def __init__(self, all_goal_joint_names, latent_dim):
        super().__init__()
        
        # todo: use a tensor of weight per joint, and another one for biases, then apply a selection in one go like we to for actions
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']
        self.extended_goal_joint_idx = {joint_name: idx for idx, joint_name in enumerate(self.extended_goal_joint_names)}
        self.n_extended_goal_joints = len(self.extended_goal_joint_names)
        self.target_loc_emb = nn.ParameterDict({joint_name: 
            nn.Sequential(
                nn.Linear(3, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim)) 
            for joint_name in self.extended_goal_joint_names})  # todo: check if 3 works for heading and traj
            # nn.Linear(3, latent_dim) for joint_name in self.extended_goal_joint_names})  # todo: check if 3 works for heading and traj
        self.target_all_loc_emb = WeightedSum(self.n_extended_goal_joints) # nn.Linear(self.n_extended_goal_joints, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, input, target_joint_names, target_heading):
        output = torch.zeros((input.shape[0], self.latent_dim), dtype=input.dtype, device=input.device)
        
        # Iterate over the batch and apply the appropriate filter for each joint
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names
            output_one_sample = torch.zeros((self.n_extended_goal_joints, self.latent_dim), dtype=input.dtype, device=input.device)
            for joint_name in sample_joint_names_w_heading:
                layer = self.target_loc_emb[joint_name]
                output_one_sample[self.extended_goal_joint_idx[joint_name]] = layer(input[sample_idx, self.extended_goal_joint_idx[joint_name]])  
            output[sample_idx] = self.target_all_loc_emb(output_one_sample)
            # print(torch.where(output_one_sample.sum(axis=1)!=0)[0].cpu().numpy())
               
        return output
