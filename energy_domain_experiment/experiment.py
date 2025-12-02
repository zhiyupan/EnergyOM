from energy_domain_experiment.utilizing import Experiment
from Model.bert_Energy_tsdae import BertEnergy
from Model.deepseek import DeepSeekMatcher
from Model.gpt_4o_mini import GPT4oMiniMatcher

energy_bert_model = BertEnergy()
deepseek_model = DeepSeekMatcher()
gpt_model = GPT4oMiniMatcher()
#exp_1 = Experiment(energy_bert_model, 4)
#exp_1.experiment_without_llm_single_directional()
# exp_2 = Experiment(energy_bert_model, 2)
# exp_2.experiment_without_llm_bidirectional()
exp_3 = Experiment(energy_bert_model, 6)
exp_3.experiment_with_llm_single_directional(f"../energy_domain_experiment/experiment_1_results", deepseek_model)
# 4.
# exp_4.experiment_with_llm_single_directional(f"../energy_domain_experiment/experiment_2_results", deepseek_model)

