from pipelines import (comapre_DAQN_vs_DQN,
                       run_DAQN_epsilon_comparator,
                       run_DAQN_gamma_comparator,
                       test_model)





if __name__ == "__main__":
    
    
    #comapre_DAQN_vs_DQN(num_runs = 100,
    #                    num_episodes=1500)
    
    
    #run_DAQN_epsilon_comparator(fix_eps = 0.1,
    #                            var_eps_start = 1.0,
    #                            var_eps_min = 0.05,
    #                            var_eps_decay = 0.995,
    #                            num_runs = 100,
    #                            num_episodes=1500)
    
    
    #run_DAQN_gamma_comparator(gamma_values=[1.0, 0.95, 0.9],
    #                          num_runs=100,
    #                          num_episodes=1500)
    
    
    test_model(model_path='trained_models/DAQN/DAQN_var_eps_model.pth', 
               test_file='DAQN/data/DAQN_test_input_format.txt')