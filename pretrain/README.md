# Pre-Training Resources 🔧

To enhance research transparency and reproducibility, we are open-sourcing relevant pre-training resources:

### Pre-Training


<details><summary>1. Pre-training and Evaluation Code</summary>

The pre-training code can be found [here](https://github.com/RUC-GSAI/YuLan-Mini/tree/main/pretrain). Note that due to subsequent code modifications, this code may not run directly and may require some adjustments.

<h3 id="key-features-">Key Features:</h3>
<ol>
<li><strong>Stability</strong>: We adopted muP initialization and scaling factor, as well as the reparameterization method of WeSaR, achieving training stability without significantly increasing training time. For details, see our technical report.</li>
<li><strong>Training efficiency</strong>: By using the <code>flash_attn</code> and <code>liger_kernel</code> libraries, we achieved 51% MFU (in comparison, Megatron only has about 41% MFU on small models of the same scale).</li>
<li><strong>Data curriculum</strong>: We modified the HF Trainer to make it suitable for training in successive curriculum phases and different decay functions of WSD.</li>
<li><strong>Other features</strong>: Support automatic restart training of torchrun, wandb records hidden states to monitor training stability, and other attempts (such as QK-LayerNorm, Embedding Gradient Shrink, etc.).</li>
</ol>

<pre><code>├── train.py  <span class="hljs-comment"># 👈🏻 The main training script</span>
├── train.<span class="hljs-keyword">sh </span> <span class="hljs-comment"># 👈🏻 The main training script for each curriculum phase</span>
├── yulanmini-2B-final-phase25.<span class="hljs-keyword">sh </span> <span class="hljs-comment"># 👈🏻 example script for phase 25</span>
├── yulanmini-2B-s25d-decay80-1sqrt-long-28k-final-phase26.<span class="hljs-keyword">sh </span> <span class="hljs-comment"># 👈🏻 example script for phase 26</span>
├── ds2_config_adamw.<span class="hljs-keyword">json </span> <span class="hljs-comment"># The DeepSpeed configuration file</span>
├── setup.<span class="hljs-keyword">sh </span> <span class="hljs-comment"># The setup script for the training environment</span>
├── torchrun_wrapper.<span class="hljs-keyword">sh </span> <span class="hljs-comment"># The wrapper script for torchrun</span>
├── train_utils.py  <span class="hljs-comment"># The training utility functions</span>
└── yulanmini_trainer.py  <span class="hljs-comment"># 👈🏻 The Trainer class for training</span>
</code></pre>

<h3 id="key-features-">Continual Training Tutorial:</h3>
<h4 id="step-1-modify-the-config-json-">Step 1: Modify the <code>trainer_state.json</code></h4>
<p>Due to the implementation of Hugging Face Trainer, certain parameters are stored in the <code>trainer_state.json</code> file and cannot be modified through the Trainer&#39;s command-line arguments. Therefore, you need to update these parameters in the <code>trainer_state.json</code> file first, particularly:</p>
<ul>
<li><strong><code>save_steps</code></strong>: The frequency of saving intermediate checkpoints.</li>
<li><strong><code>train_batch_size</code></strong>: The batch size per GPU (equivalent to <code>per_device_train_batch_size</code> in the Trainer). We used a batch size of 1008 (approximately 4M tokens) during the stable training stage. Maintaining this same batch size is equally important for training effectiveness.</li>
</ul>
<p>Below is an example of a properly configured <code>trainer_state.json</code> file:</p>
<pre><code class="lang-json">{
  <span class="hljs-attr">"best_metric"</span>: <span class="hljs-literal">null</span>,
  <span class="hljs-attr">"best_model_checkpoint"</span>: <span class="hljs-literal">null</span>,
  <span class="hljs-attr">"epoch"</span>: <span class="hljs-number">0.0</span>,
  <span class="hljs-attr">"eval_steps"</span>: <span class="hljs-number">500</span>,
  <span class="hljs-attr">"global_step"</span>: <span class="hljs-number">0</span>,
  <span class="hljs-attr">"is_hyper_param_search"</span>: <span class="hljs-literal">false</span>,
  <span class="hljs-attr">"is_local_process_zero"</span>: <span class="hljs-literal">true</span>,
  <span class="hljs-attr">"is_world_process_zero"</span>: <span class="hljs-literal">true</span>,
  <span class="hljs-attr">"log_history"</span>: [],
  <span class="hljs-attr">"logging_steps"</span>: <span class="hljs-number">3</span>,
  <span class="hljs-attr">"max_steps"</span>: <span class="hljs-number">0</span>,
  <span class="hljs-attr">"num_input_tokens_seen"</span>: <span class="hljs-number">0</span>,
  <span class="hljs-attr">"num_train_epochs"</span>: <span class="hljs-number">0</span>,
  <span class="hljs-attr">"save_steps"</span>: <span class="hljs-number">250</span>,
  <span class="hljs-attr">"stateful_callbacks"</span>: {
    <span class="hljs-attr">"TrainerControl"</span>: {
      <span class="hljs-attr">"args"</span>: {
        <span class="hljs-attr">"should_epoch_stop"</span>: <span class="hljs-literal">false</span>,
        <span class="hljs-attr">"should_evaluate"</span>: <span class="hljs-literal">false</span>,
        <span class="hljs-attr">"should_log"</span>: <span class="hljs-literal">false</span>,
        <span class="hljs-attr">"should_save"</span>: <span class="hljs-literal">true</span>,
        <span class="hljs-attr">"should_training_stop"</span>: <span class="hljs-literal">true</span>
      },
      <span class="hljs-attr">"attributes"</span>: {}
    }
  },
  <span class="hljs-attr">"total_flos"</span>: <span class="hljs-number">0</span>,
  <span class="hljs-attr">"train_batch_size"</span>: <span class="hljs-number">3</span>,
  <span class="hljs-attr">"trial_name"</span>: <span class="hljs-literal">null</span>,
  <span class="hljs-attr">"trial_params"</span>: <span class="hljs-literal">null</span>
}
</code></pre>
<h4 id="step-2-enable-universal-checkpointing-in-the-deepspeed-configuration">Step 2: Enable Universal Checkpointing in the DeepSpeed Configuration</h4>
<p>To ensure DeepSpeed Integration loads the Universal Checkpoint, you need to enable this feature in the DeepSpeed configuration JSON file. </p>
<p>Here is an example of a ZeRO2 configuration with Universal Checkpointing enabled:</p>
<pre><code class="lang-json">{
  <span class="hljs-attr">"bf16"</span>: {
    <span class="hljs-attr">"enabled"</span>: <span class="hljs-string">"auto"</span>
  },
  <span class="hljs-attr">"zero_optimization"</span>: {
    <span class="hljs-attr">"stage"</span>: <span class="hljs-number">2</span>,
    <span class="hljs-attr">"allgather_partitions"</span>: <span class="hljs-literal">true</span>,
    <span class="hljs-attr">"allgather_bucket_size"</span>: <span class="hljs-number">8e8</span>,
    <span class="hljs-attr">"overlap_comm"</span>: <span class="hljs-literal">true</span>,
    <span class="hljs-attr">"reduce_scatter"</span>: <span class="hljs-literal">true</span>,
    <span class="hljs-attr">"reduce_bucket_size"</span>: <span class="hljs-number">8e8</span>,
    <span class="hljs-attr">"contiguous_gradients"</span>: <span class="hljs-literal">true</span>
  },
  <span class="hljs-attr">"gradient_accumulation_steps"</span>: <span class="hljs-string">"auto"</span>,
  <span class="hljs-attr">"gradient_clipping"</span>: <span class="hljs-string">"auto"</span>,
  <span class="hljs-attr">"steps_per_print"</span>: <span class="hljs-number">16</span>,
  <span class="hljs-attr">"train_batch_size"</span>: <span class="hljs-string">"auto"</span>,
  <span class="hljs-attr">"train_micro_batch_size_per_gpu"</span>: <span class="hljs-string">"auto"</span>,
  <span class="hljs-attr">"wall_clock_breakdown"</span>: <span class="hljs-literal">false</span>,
  <span class="hljs-attr">"dump_state"</span>: <span class="hljs-literal">true</span>,
  <span class="hljs-attr">"optimizer"</span>: {
    <span class="hljs-attr">"type"</span>: <span class="hljs-string">"AdamW"</span>,
    <span class="hljs-attr">"params"</span>: {
      <span class="hljs-attr">"lr"</span>: <span class="hljs-string">"auto"</span>,
      <span class="hljs-attr">"betas"</span>: <span class="hljs-string">"auto"</span>,
      <span class="hljs-attr">"eps"</span>: <span class="hljs-string">"auto"</span>,
      <span class="hljs-attr">"weight_decay"</span>: <span class="hljs-string">"auto"</span>
    }
  },
  <span class="hljs-attr">"checkpoint"</span>: {
    <span class="hljs-attr">"load_universal"</span>: <span class="hljs-literal">true</span>
  }
}
</code></pre>
<h4 id="step-3-resume-training">Step 3: Resume Training</h4>
<p>When calling <code>trainer.train</code>, include the <code>resume_from_checkpoint</code> argument to load the distributed optimizer state from the Universal Checkpoint and resume training.</p>
<pre><code class="lang-python"><span class="hljs-attr">trainer.train(resume_from_checkpoint</span>=<span class="hljs-string">training_args.resume_from_checkpoint)</span>
</code></pre>
<p>We provide an internal <a href="https://github.com/RUC-GSAI/YuLan-Mini/tree/main/pretrain">training framework</a> for your reference, but you are free to choose other frameworks.</p>

</details>

<details><summary>2. Intermediate Stage Checkpoints</summary>
The intermediate stage checkpoints are released in <a href="https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3">YuLan-Mini</a>.

<table>
    <thead>
        <tr>
            <th>Stage</th>
            <th>Curriculum Phase</th>
            <th>4K Context</th>
            <th>28K Context</th>
            <th>Optimizer</th>
            <th>Inference Architecture</th>
            <th>LAMBADA <code>Acc</code></th>
            <th>GSM8K <code>Acc</code></th>
            <th>HumanEval <code>pass@1</code></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Stable</td>
            <td>5</td>
            <td><a href="https://huggingface.co/yulan-team/YuLan-Mini-Phase5">YuLan-Mini-Phase5</a></td>
            <td></td>
            <td></td>
            <td><code>yulanmini</code></td>
            <td>53.85</td>
            <td>3.41</td>
            <td>12.26</td>
        </tr>
        <tr>
            <td>Stable</td>
            <td>10</td>
            <td><a href="https://huggingface.co/yulan-team/YuLan-Mini-Phase10">YuLan-Mini-Phase10</a></td>
            <td></td>
            <td></td>
            <td><code>yulanmini</code></td>
            <td>55.00</td>
            <td>9.57</td>
            <td>15.95</td>
        </tr>
        <tr>
            <td>Stable</td>
            <td>15</td>
            <td><a href="https://huggingface.co/yulan-team/YuLan-Mini-Phase15">YuLan-Mini-Phase15</a></td>
            <td></td>
            <td></td>
            <td><code>yulanmini</code></td>
            <td>55.81</td>
            <td>13.81</td>
            <td>16.99</td>
        </tr>
        <tr>
            <td>Stable</td>
            <td>20</td>
            <td><a href="https://huggingface.co/yulan-team/YuLan-Mini-Phase20">YuLan-Mini-Phase20</a></td>
            <td></td>
            <td>✅</td>
            <td><code>yulanmini</code></td>
            <td>55.81</td>
            <td>21.39</td>
            <td>20.79</td>
        </tr>
        <tr>
            <td>Stable</td>
            <td>25 (1T tokens)</td>
            <td><a href="https://huggingface.co/yulan-team/YuLan-Mini-Before-Annealing">YuLan-Mini-Before-Annealing</a></td>
            <td></td>
            <td>✅</td>
            <td><code>yulanmini</code></td>
            <td>55.67</td>
            <td>29.94</td>
            <td>34.06</td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>Annealing</td>
            <td>26</td>
            <td>YuLan-Mini-4K</td>
            <td></td>
            <td></td>
            <td><code>llama</code>*</td>
            <td>64.72</td>
            <td>66.65</td>
            <td>61.60</td>
        </tr>
        <tr>
            <td>Annealing</td>
            <td>27</td>
            <td></td>
            <td><a href="https://huggingface.co/yulan-team/YuLan-Mini">YuLan-Mini</a></td>
            <td></td>
            <td><code>llama</code>*</td>
            <td>65.67</td>
            <td>68.46</td>
            <td>64.00</td>
        </tr>
    </tbody>
</table>

\*: For easier inference and deployment, we merged the re-parameterized added parameters and scaling factors into the final released models ([**YuLan-Mini**](https://huggingface.co/yulan-team/YuLan-Mini) and **YuLan-Mini-Intermediate-4K**), enabling it to run on the Llama architecture. However, these parameters are still retained in the intermediate checkpoints from the training process.

</details>

<details><summary>3. Optimizer States Before Annealing</summary>

<a href="https://huggingface.co/yulan-team/YuLan-Mini-Before-Annealing">YuLan-Mini-Before-Annealing</a>
</details>

### Datasets


<details><summary>4. The Used Open-Source Datasets </summary>

<a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain/datasets">Used-Datasets-List</a>

</details>

<details><summary>5. Data Distribution for every phase</summary>

<a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain/datasets/final.pdf">
  <div align=center>
    <img src="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/assets/data_distribution_for_every_phase.png">
  </div>
</a>

</details>

<details><summary>6. Data Preprocessing and Synthesis Pipeline</summary>

The synthetic data we are using is released in <a href="https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3">YuLan-Mini-Datasets</a>


We also released the <a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain/preprocess">data preprocessing</a> (including data formatting, filtering, tokenization, and mixing) and <a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain/synthesis">synthesis pipeline</a> for your reference.


<div align=center>
<img src="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/assets/data-pipeline.png">
</div>

</details>


### What you can do with these pre-training resources

1. **Pre-train** your own LLM. You can use [our data](https://huggingface.co/yulan-team/YuLan-Mini-Datasets) and curriculum to train a model that's just as powerful as YuLan-Mini.
2. Perform your own **learning rate annealing**. During the annealing phase, YuLan-Mini's learning ability is at its peak. You can resume training from [the checkpoint before annealing](https://huggingface.co/yulan-team/YuLan-Mini-Before-Annealing) and use your own dataset for learning rate annealing.
3. **Fine-tune** the Instruct version of the LLM. You can use the [YuLan-Mini](https://huggingface.co/yulan-team/YuLan-Mini) base model to train your own Instruct version.
4. **Training dynamics** research. You can use YuLan-Mini's [intermediate checkpoints](https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3) to explore internal changes during the pre-training process.
5. **Synthesize** your own data. You can use YuLan-Mini's [data pipeline](https://github.com/RUC-GSAI/YuLan-Mini) to clean and generate your own dataset.

