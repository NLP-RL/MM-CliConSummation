## Experience and Evidence are the eyes of an excellent summarizer! Towards Knowledge Infused Multi-modal Clinical Conversation Summarization

The repository contains code and dataset for research article titled 'Experience and Evidence are the eyes of an excellent summarizer! Towards Knowledge Infused Multi-modal Clinical Conversation Summarization' published at 32nd ACM International Conference on Information and Knowledge Management (CIKM ’23).

### Abstract
With the advancement of telemedicine, both researchers and medical practitioners are working hand-in-hand to develop various techniques to automate various medical operations, such as diagnosis report generation. In this paper, we first present a multimodal clinical conversation summary generation task that takes a clinician-patient interaction (both textual and visual information) and generates a succinct synopsis of the conversation. We propose a knowledge-infused, multi-modal, multi-tasking medical domain identification and clinical conversation summary generation (MMCliConSummation) framework. It leverages an adapter to infuse knowledge and visual features and unify the fused feature vector using a gated mechanism. Furthermore, we developed a multi-modal, multi-intent clinical conversation summarization corpus annotated with intent, symptom, and summary. The extensive set of experiments, both quantitatively and qualitatively, led to the following findings: (a) critical significance of visuals, (b) more precise and medical entity preserving summary with additional knowledge infusion, and (c) a correlation between medical department identification and clinical synopsis generation.

![Working](https://github.com/NLP-RL/MM-CliConSummation/blob/main/MMCliConSummation.jpg)

#### Full Paper: https://arxiv.org/abs/2309.15739

### Experiment

#### Please create a new environment for the dependencies using the following command:

	conda env create -f environment.yml

#### Activate conda environment after installation by using the command:

	conda activate environment
	
#### PreProcessing:

	 python pre_processing.py (Inside Data folder)
   
#### Training and Testing:
 
 ##### For Overall Summary, please go to folder named MM-CCS
  
    python MM-CliConSummation.py
    
 ##### For Overall Summary, please go to folder named MCS
    
    python MM-CliConSummation.py
    
 ##### For Ablation Study, please go to folder named AS

    python L32.py (visual feature at layer 3 and knolwdge feature at 2)

### Full Dataset Access

We provide the dataset for research and academic purposes. To request access to the dataset, please follow the instructions below:

1. **Fill Out the Request Form**: To access the corpus, you need to submit a request through our [Google Form](https://forms.gle/C5q7jDprPGsCuYcD6).

2. **Review and Approval**: After submitting the form, your request will be reviewed. If approved, you will receive an email with a link to download the dataset.

3. **Terms of Use**: By requesting access, you agree to:
    - Use the dataset solely for non-commercial, educational, and research purposes.
    - Not use the dataset for any commercial activities.
    - Attribute the creators of this resource in any works (publications, presentations, or other public dissemination) utilizing the dataset.
    - Not disseminate the dataset without prior permission from the appropriate authorities.

### Citation information:
If you find this code useful in your research, please consider citing:
~~~~
@inproceedings{tiwari2023experience,
  title={Experience and Evidence are the eyes of an excellent summarizer! Towards Knowledge Infused Multi-modal Clinical Conversation Summarization},
  author={Tiwari, Abhisek and Saha, Anisha and Saha, Sriparna and Bhattacharyya, Pushpak and Dhar, Minakshi},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={2452--2461},
  year={2023}
}

Please contact us @ abhisektiwari2014@gmail.com for any questions, suggestions, or remarks.
