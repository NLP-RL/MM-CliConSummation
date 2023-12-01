# MM-CliConSummation

Towards Knowledge Infused Multi-modal Clinical Conversation Summarization

### Please create a new environment for the dependencies using the following command:

	conda env create -f environment.yml

### Activate conda environment after installation by using the command:

	conda activate environment
	
### PreProcessing:

	 python pre_processing.py (Inside Data folder)
   
### Training and Testing:
 
 #### For Overall Summary, please go to folder named MM-CCS
  
    python MM-CliConSummation.py
    
####  For Overall Summary, please go to folder named MCS
    
    python MM-CliConSummation.py
    
#### For Ablation Study, please go to folder named AS

    python L32.py (visual feature at layer 3 and knolwdge feature at 2)

### Citation information:

@inproceedings{tiwari2023experience,
  title={Experience and Evidence are the eyes of an excellent summarizer! Towards Knowledge Infused Multi-modal Clinical Conversation Summarization},
  author={Tiwari, Abhisek and Saha, Anisha and Saha, Sriparna and Bhattacharyya, Pushpak and Dhar, Minakshi},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={2452--2461},
  year={2023}
}

    
