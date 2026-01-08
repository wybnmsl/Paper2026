
# class Paras():
#     def __init__(self):
#         #####################
#         ### General settings  ###
#         #####################
#         self.method = 'eoh'
#         self.problem = 'tsp_construct'
#         self.selection = None
#         self.management = None

#         #####################
#         ###  EC settings  ###
#         #####################
#         self.ec_pop_size = 5  # number of algorithms in each population, default = 10
#         self.ec_n_pop = 5 # number of populations, default = 10
#         self.ec_operators = None # evolution operators: ['e1','e2','m1','m2'], default =  ['e1','e2','m1','m2']
#         self.ec_m = 2  # number of parents for 'e1' and 'e2' operators, default = 2
#         self.ec_operator_weights = None  # weights for operators, i.e., the probability of use the operator in each iteration, default = [1,1,1,1]
        
#         #####################
#         ### LLM settings  ###
#         #####################
#         self.llm_use_local = False  # if use local model
#         self.llm_local_url = None  # your local server 'http://127.0.0.1:11012/completions'
#         self.llm_api_endpoint = None # endpoint for remote LLM, e.g., api.deepseek.com
#         self.llm_api_key = None  # API key for remote LLM, e.g., sk-xxxx
#         self.llm_model = None  # model type for remote LLM, e.g., deepseek-chat

#         #####################
#         ###  Exp settings  ###
#         #####################
#         self.exp_debug_mode = False  # if debug
#         self.exp_output_path = "./"  # default folder for ael outputs
#         self.exp_use_seed = False
#         self.exp_seed_path = "./seeds/seeds.json"
#         self.exp_use_continue = False
#         self.exp_continue_id = 0
#         self.exp_continue_path = "./results/pops/population_generation_0.json"
#         self.exp_n_proc = 1
        
#         #####################
#         ###  Evaluation settings  ###
#         #####################
#         self.eva_timeout = 30
#         self.eva_numba_decorator = False

#         #####################
#         ###  DAG/Planner knobs  ###
#         #####################
#         self.use_dag_runtime = True     # 是否用 DAG 执行器包装问题求解（若问题侧实现了）
#         self.expose_profile  = True     # 是否把 profile 往上层 other_inf 回传
#         self.use_llm_dag_planner = False  # 可选：是否启用 LLM 产出 DAG（问题侧可读取）


#     def set_parallel(self):
#         import multiprocessing
#         num_processes = multiprocessing.cpu_count()
#         if self.exp_n_proc == -1 or self.exp_n_proc > num_processes:
#             self.exp_n_proc = num_processes
#             print(f"Set the number of proc to {num_processes} .")
    
#     def set_ec(self):    
        
#         if self.management == None:
#             if self.method in ['ael','eoh']:
#                 self.management = 'pop_greedy'
#             elif self.method == 'ls':
#                 self.management = 'ls_greedy'
#             elif self.method == 'sa':
#                 self.management = 'ls_sa'
        
#         if self.selection == None:
#             self.selection = 'prob_rank'
            
        
#         if self.ec_operators == None:
#             if self.method == 'eoh':
#                 self.ec_operators  = ['e1','e2','m1','m2']
#             elif self.method == 'ael':
#                 self.ec_operators  = ['crossover','mutation']
#             elif self.method == 'ls':
#                 self.ec_operators  = ['m1']
#             elif self.method == 'sa':
#                 self.ec_operators  = ['m1']

#         # if self.ec_operator_weights == None:
#         #     self.ec_operator_weights = [1 for _ in range(len(self.ec_operators))]
#         # elif len(self.ec_operator) != len(self.ec_operator_weights):
#         #     print("Warning! Lengths of ec_operator_weights and ec_operator shoud be the same.")
#         #     self.ec_operator_weights = [1 for _ in range(len(self.ec_operators))]

#         if self.ec_operator_weights is None:
#             self.ec_operator_weights = [1 for _ in range(len(self.ec_operators))]
#         elif len(self.ec_operators) != len(self.ec_operator_weights):
#             print("Warning! Lengths of ec_operator_weights and ec_operators should be the same. Reset to all-ones.")
#             self.ec_operator_weights = [1 for _ in range(len(self.ec_operators))]

        
                    
#         if self.method in ['ls','sa'] and self.ec_pop_size >1:
#             self.ec_pop_size = 1
#             self.exp_n_proc = 1
#             print("> single-point-based, set pop size to 1. ")
            
#     def set_evaluation(self):
#         # Initialize evaluation settings
#         if self.problem == 'bp_online':
#             self.eva_timeout = 20
#             self.eva_numba_decorator  = True
#         elif self.problem == 'tsp_construct':
#             self.eva_timeout = 20
                
#     def set_paras(self, *args, **kwargs):
        
#         # Map paras
#         for key, value in kwargs.items():
#             if hasattr(self, key):
#                 setattr(self, key, value)
              
#         # Identify and set parallel 
#         self.set_parallel()
        
#         # Initialize method and ec settings
#         self.set_ec()
        
#         # Initialize evaluation settings
#         self.set_evaluation()




# if __name__ == "__main__":

#     # Create an instance of the Paras class
#     paras_instance = Paras()

#     # Setting parameters using the set_paras method
#     paras_instance.set_paras(llm_use_local=True, llm_local_url='http://example.com', ec_pop_size=8)

#     # Accessing the updated parameters
#     print(paras_instance.llm_use_local)  # Output: True
#     print(paras_instance.llm_local_url)  # Output: http://example.com
#     print(paras_instance.ec_pop_size)    # Output: 8
            
            
            
# eoh/src/eoh/utils/getParas.py
class Paras():
    def __init__(self):
        #####################
        ### General settings  ###
        #####################
        self.method = 'eoh'
        self.problem = 'tsp_construct'
        self.selection = None
        self.management = None

        #####################
        ###  EC settings  ###
        #####################
        self.ec_pop_size = 5
        self.ec_n_pop = 5
        self.ec_operators = None
        self.ec_m = 2
        self.ec_operator_weights = None
        
        #####################
        ### LLM settings  ###
        #####################
        self.llm_use_local = False
        self.llm_local_url = None
        self.llm_api_endpoint = None
        self.llm_api_key = None
        self.llm_model = None

        #####################
        ###  Exp settings  ###
        #####################
        self.exp_debug_mode = False
        self.exp_output_path = "./"
        self.exp_use_seed = False
        self.exp_seed_path = "./seeds/seeds.json"
        self.exp_use_continue = False
        self.exp_continue_id = 0
        self.exp_continue_path = "./results/pops/population_generation_0.json"
        self.exp_n_proc = 1
        
        #####################
        ###  Evaluation settings  ###
        #####################
        self.eva_timeout = 30
        self.eva_numba_decorator = False

        #####################
        ###  DAG/Planner knobs  ###
        #####################
        self.use_dag_runtime = True
        self.expose_profile  = True
        self.use_llm_dag_planner = False

        #####################
        ###  T-phase thresholds/switches  ###
        #####################
        self.t_alpha = {"t1":0.20,"t2":0.10,"t3":0.05}
        self.t_beta_abs = {"t1":1.0,"t2":0.8,"t3":0.5}
        self.t_gamma_rel = 0.10
        self.t_gamma_abs = 0.5
        self.t_Omax = 5.0
        self.t_bypass_on_fail = True
        self.t_diag_retry = False
        self.t_verbose = True

    def set_parallel(self):
        import multiprocessing
        num_processes = multiprocessing.cpu_count()
        if self.exp_n_proc == -1 or self.exp_n_proc > num_processes:
            self.exp_n_proc = num_processes
            print(f"Set the number of proc to {num_processes} .")
    
    def set_ec(self):    
        if self.management == None:
            if self.method in ['ael','eoh']:
                self.management = 'pop_greedy'
            elif self.method == 'ls':
                self.management = 'ls_greedy'
            elif self.method == 'sa':
                self.management = 'ls_sa'
        
        if self.selection == None:
            self.selection = 'prob_rank'
            
        if self.ec_operators == None:
            if self.method == 'eoh':
                self.ec_operators  = ['e1','e2','m1','m2']
            elif self.method == 'ael':
                self.ec_operators  = ['crossover','mutation']
            elif self.method == 'ls':
                self.ec_operators  = ['m1']
            elif self.method == 'sa':
                self.ec_operators  = ['m1']

        if self.ec_operator_weights is None:
            self.ec_operator_weights = [1 for _ in range(len(self.ec_operators))]
        elif len(self.ec_operators) != len(self.ec_operator_weights):
            print("Warning! Lengths of ec_operator_weights and ec_operators should be the same. Reset to all-ones.")
            self.ec_operator_weights = [1 for _ in range(len(self.ec_operators))]

        if self.method in ['ls','sa'] and self.ec_pop_size >1:
            self.ec_pop_size = 1
            self.exp_n_proc = 1
            print("> single-point-based, set pop size to 1. ")
            
    def set_evaluation(self):
        if self.problem == 'bp_online':
            self.eva_timeout = 20
            self.eva_numba_decorator  = True
        elif self.problem == 'tsp_construct':
            self.eva_timeout = 20
                
    def set_paras(self, *args, **kwargs):
        # Map paras
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
              
        # Identify and set parallel 
        self.set_parallel()
        
        # Initialize method and ec settings
        self.set_ec()
        
        # Initialize evaluation settings
        self.set_evaluation()


if __name__ == "__main__":
    paras_instance = Paras()
    paras_instance.set_paras(llm_use_local=True, llm_local_url='http://example.com', ec_pop_size=8)
    print(paras_instance.llm_use_local)
    print(paras_instance.llm_local_url)
    print(paras_instance.ec_pop_size)
