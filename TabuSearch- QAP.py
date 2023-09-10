# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 00:43:47 2020

@author: mary
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:11:26 2020

@author: maryam
"""
#Tabu Search 
# 1)  encode the problem as permutation,
# 2) define a neighborhood and a move operator
# 3) set a tabu list size and select a stopping criterion. 
# 4) Use only a recency based tabu list and no aspiration criteria at this point.

#%%
#import libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random 
np.random.seed(777)

#%%
# fixed tabu length 
len_tabu = 20

#dynamic tabu list 
#len_tabu = random.randint(7, 20)

N = 105 
gens = 1 
len_sol = 15
int_solution = ['D2','D1','D3','D4','D13', 'D6','D7', 'D8', 'D10','D9', 'D11','D12','D5','D14','D15']

#%%
# import distance matrix from excel
# remove first column and row indices 
dist_flow_mat = pd.read_csv('distflow_hw5.csv', header = None)

#%%
distflow_mat = dist_flow_mat.to_numpy()

#%%
#a = np.matrix([[11,19, 6],[10,5,7], [3, 2, 4]])
#dist_mat = dist_mat + dist_mat.T - np.diag(np.diag(dist_mat))
#print(dist_mat)

#get the upper triangular part of this matrix
distflow_uppertr = distflow_mat[np.triu_indices(distflow_mat.shape[0], k = 0)]
# [1 2 3 5 6 9]

# put it back into a 2D symmetric array
size_dist= 15
distflow_mat = np.zeros((size_dist,size_dist))
distflow_mat[np.triu_indices(distflow_mat.shape[0], k = 0)]= distflow_uppertr 
dist_mat = distflow_mat + distflow_mat.T - np.diag(np.diag(distflow_mat))
#array([[1., 2., 3.],
#       [2., 5., 6.],
#       [3., 6., 9.]])

#%%  lower matrix
distflow_matf = dist_flow_mat.to_numpy()
distflow_lowertr = np.tril(distflow_matf ,k=-1)

size_flow = 15
#z = np.zeros((size_flow,size_flow))

flow_mat = distflow_lowertr + distflow_lowertr.T - np.diag(np.diag(distflow_lowertr))

#%%
flow_matrix = pd.DataFrame(flow_mat)

flow_matrix = flow_matrix.rename(index={0: "D1", 1: "D2", 2: "D3", 3: "D4",
                                          4: "D5", 5: "D6", 6: "D7", 7: "D8",
                                          8: "D9", 9: "D10", 10: "D11", 
                                          11: "D12", 12: "D13", 13: "D14", 14: "D15"})

flow_matrix.columns = ['D1','D2','D3','D4','D5', 'D6','D7', 'D8', 'D9','D10', 'D11','D12','D13','D14','D15']


#%%
dist_matrix =  pd.DataFrame(dist_mat)

dist_matrix = dist_matrix.rename(index={0: "D1", 1: "D2", 2: "D3", 3: "D4",
                                          4: "D5", 5: "D6", 6: "D7", 7: "D8",
                                          8: "D9", 9: "D10", 10: "D11", 
                                          11: "D12", 12: "D13", 13: "D14", 14: "D15"})
dist_matrix.columns = ["D1",'D2','D3','D4','D5', 'D6','D7', 'D8', 'D9','D10', 'D11','D12', 'D13','D14','D15']

#%%
int_solution = ['D2','D14','D1','D4','D13', 'D6','D8', 'D10', 'D7','D9', 'D12','D11','D5','D3','D15']

#%% calaculatr objective function 
new_dist_matrix = dist_matrix.reindex (index = int_solution , columns = int_solution)
new_dist= new_dist_matrix.to_numpy()
new_flowcost = pd.DataFrame(new_dist*flow_matrix)
new_fc = np.array(new_flowcost)
sum_flowcost = sum(sum(new_fc))

#%%
def flowcost (initial):
    
    new_dist_matrix = dist_matrix.reindex (index = initial, columns = initial)
    new_dist= new_dist_matrix.to_numpy()
    new_flowcost = pd.DataFrame(new_dist*flow_matrix)
    new_fc = np.array(new_flowcost)
    sum_flowcost = sum(sum(new_fc))
    return sum_flowcost
#%% 
def Convert(tup, di):
    di = dict(tup)
    return di 

#%%
dic_swap = {}

int_solution = ['D2','D14','D1','D4','D13', 'D6','D8', 'D10', 'D7','D9', 'D12','D11','D5','D3','D15']
import itertools as itr
row_rep = 0
row_rep_not = 0
tabu_list =  np.empty((0,len(int_solution)))
# move operator is 2-opt
#number of elements in the swap_list is equal to 105 
df_iter = pd.DataFrame()
all_solutions=np.empty((0,15))

best_so_far={}
for i in range(10): 
   

    df_cost = pd.DataFrame(columns = ['cost'])
    df_swap_= pd.DataFrame()
    
    
#    print(int_solution)
#    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    #if i>0:
     #   print(df_swap_.iloc[0, 0:15].loc[df_swap_['iteration'] == i-1])

    swap_list = list(itr.combinations(int_solution, 2))
# all permutations
#    from itertools import permutations
#for i in range(runs):
#    perm = permutations(int_solution, 2) 
#    swap_list = list(perm)
    counter = 0    
    
    for k in swap_list:
        
        swap_counter = swap_list[counter]

        f_swap = swap_counter[0]
        s_swap = swap_counter[1]
        
        time.sleep(0)
#        print(f_swap, s_swap)
#        print('------------------------------------------------------------------------')
        
        swap=[]
        count = 0
        cur_sol = int_solution
        print(swap_list[counter][0], swap_list[counter][1])
       
        for j in cur_sol: 
            if cur_sol[count] == f_swap:
                 swap = np.append(swap,s_swap)
                
            elif cur_sol[count] == s_swap:
                 swap = np.append(swap,f_swap)
                
            else:
                swap= np.append(swap,cur_sol[count])
                
            count += 1 


#
        #print(swap.shape)
        dic_temp = {}
        cost_swap = flowcost(cur_sol)
        dic_temp[cost_swap] = cur_sol
        #print(dic_swap)
       # print('=========================')
        dictionary = {}
        dic_temp = Convert(dic_temp, dictionary) 
        dic_temp =pd.DataFrame.from_dict(dic_temp, orient='index')
        #dic_temp.columns = ['C1','C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11','C12', 'C13', 'C14', 'C15']
        
#        print('***********************************************************', f_swap, s_swap)        
#        print(dic_temp)
#        print('***********************************************************')
#        print(cur_sol)
#        print('====================================================')
#        print('====================================================')
        time.sleep(0)
        df_swap_ = df_swap_.append(dic_temp, ignore_index= True)
        df_cost.at[counter, 'cost'] = cost_swap
        
        counter +=1
    
    df_swap_['iteration'] =  i
    df_swap_['cost'] = df_cost['cost']
    df_swap_ = df_swap_.sort_values('cost') 
    df_swap_ = df_swap_.reset_index()
    print(df_swap_)
    df_swap_ = df_swap_[[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 'cost', 'iteration']]
    #print(df_swap_)
#    #%%
#    df_swap_.loc[df_swap_['iteration'] == i].iloc[0]
#    #%%
#    df_swap_['cost'].loc[df_swap_['iteration'] == i].iloc[0]
#    #%%
#    int_solution = df_swap_.loc[df_swap_['iteration'] == i].iloc[0, 0:-2].tolist()
    #print(df_swap_)
    time.sleep(0)

    solution = df_swap_.loc[df_swap_['iteration'] == i].iloc[0, 0:-2].to_list()
    
    full_str = ''.join([str(elem) for elem in solution ])
    best = df_swap_.loc[df_swap_['iteration'] == i].iloc[0, 0:-2].tolist()
    time.sleep(0)
    print('best', best)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(solution, 'solution', df_swap_['cost'][0])
    
    
    df_iter = df_iter.append(df_swap_.loc[df_swap_['iteration'] == i].iloc[0, :])
    
#    if i==0:
#        ex_cost = flowcost(best_soln) 
    initial = ['D2','D14','D1','D4','D13', 'D6','D8', 'D10', 'D7','D9', 'D12','D11','D5','D3','D15']
    fc_costs = []
    fc_costs.append(flowcost(initial))
    #iter=0



    cur_sol =  df_swap_['cost'][0]
    
    while df_swap_['cost'][0] in df_cost: 
          cur_sol = df_swap_['cost'][0]

    if len(tabu_list) > len_tabu:
        del tabu_list[0]
    
    
    tabu_list = np.vstack((solution, tabu_list))
    all_solutions = np.vstack((solution,all_solutions))

    for a in all_solutions:
        
        if (flowcost(a)) <= min(fc_costs):
            fc_costs.append(flowcost(solution))
            list_a = a.tolist()
            best_so_far[i] = list_a
            print('********************')
            time.sleep(2)
            print('bestsofar', best_so_far[i])
            
        
    
#%%    
    
    
    
    
    
    
    
    
    
    
    
    
    #555555555555555555555555555555555555555555555555555555555555555555555
    
    if i == 0 and solution not in tabu_list:
        best_soln = initial
        print('bestsol', solution)
        if flowcost(solution) <= flowcost(best_soln):
            fc_costs.append(flowcost(solution))
            solution_ar = np.asarray(solution)
            tabu_list = np.vstack((solution_ar, tabu_list))      
                
        else:
           
           solution_ar = np.asarray(solution)
           tabu_list = np.vstack((solution_ar, tabu_list))
           print('tabulist', tabu_list)
           if len(tabu_list) > len_tabu:
                del tabu_list[0]
                
           if flowcost(solution) <= flowcost(best_soln):
                fc_costs.append(flowcost(solution))
                
                
    elif solution not in tabu_list: 
        flowcost(solution) <= flowcost(best_soln)
        best_soln = solution
        fc_costs.append(flowcost(solution))
        solution_ar = np.asarray(solution)
        tabu_list = np.vstack((solution_ar, tabu_list))
        print('tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt')
        print('tabulst', tabu_list)
        print('tabulst', tabu_list)
        print('cost', fc_costs)
        if len(tabu_list) > len_tabu:
            del tabu_list[0]
            print('ddddddddddddddddddddddddddddeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeelllllllllllll')
            print('tabulst', tabu_list)
                
    else:
            if flowcost(solution) <= flowcost(best_soln):
                fc_costs.append(flowcost(solution))
                print('iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiintttttttttttttttttttttttttaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbuuuuuuuuuuu')
                solution_ar = np.asarray(solution)
                tabu_list = np.vstack((solution_ar, tabu_list))
                print('tabulist', tabu_list)
                if len(tabu_list) > len_tabu:
                    del tabu_list[0]
            
    print('********************')
    time.sleep(2)
    
    tabu_list = np.vstack((solution, tabu_list))
    all_solutions = np.vstack((solution,all_solutions))

    for a in all_solutions:
        
        if (flowcost(a)) <= min(fc_costs):
            list_a = a.tolist()
            best_so_far[i] = list_a
            int_solution = best_so_far[i]
            print('********************')
            time.sleep(2)
            print('bestsofar', best_so_far[i])
            
            
        
       
        

#%%      

        
        
    else: 
        best_soln = solution
        if solution not in tabu_list: 
         if flowcost(solution) <= flowcost(int_solution):
            fc_costs.append(flowcost(solution))
            solution_ar = np.asarray(solution)
            tabu_list.append(solution_ar)
            
            if len(tabu_list) > len_tabu:
                del tabu_list[0]
        
            
        else:
           print('==========================================================================================================================================')
           solution_ar = np.asarray(solution)
           tabu_list.append(solution_ar)
           print('tabulist', tabu_list)
           if len(tabu_list) > len_tabu:
                del tabu_list[0]
                
           if flowcost(solution) <= flowcost(int_solution):
                fc_costs.append(flowcost(solution))
                print('********************')
                print(solution)
                print(tabu_list)
                #print(in_nested_list(tabu_list, solution))
                print('********************')
                time.sleep(0)
        
            
 

#%%    
     x = [1, 3, [1, 2, 3], [2, 3, 4], [3, 4, [], [2, 3, 'a']]]
     print (in_nested_list(x, [1, 2, 4]))  
   #%%     
def in_nested_list(my_list, item):
    """
    Determines if an item is in my_list, even if nested in a lower-level list.
    """
    if item in my_list:
        return True
    else:
        return any(in_nested_list(sublist, item) for sublist in my_list if isinstance(sublist, list))
        
        
        #%%
        
        if  df_swap_['cost'][0] < ex_cost:
            best_soln = solution
            
    ex_cost = df_swap_['cost'][0]               
               
    print('--------------------------------------------------------------------------------------')            
    print('tabulist', tabu_list)
    print('bestsol',best_soln)
    print(df_swap_['cost'][i],flowcost(solution),df_swap_['cost'][0] < flowcost(best_soln) )
    print('--------------------------------------------------------------------------------------')  
    time.sleep(2.5)
            
#%%            
            
            
    elif df_swap_['cost'][i] < flowcost(best_soln):
            tabu_list.append(solution)
            
            if len(tabu_list) > len_tabu:
                del tabu_list[0]
                
                best_soln = solution
 
        
#%%        
solu = [1,2,3,4] 
tabu = []

if solu not in tabu: 
    tabu.append(solu)

 #%%  
    


        df_iter = df_iter.append(df_swap_.loc[df_swap_['iteration'] == i].iloc[0, :])
        df_iter['tablist'] =  df_iter.apply(lambda row: row[0 : 15].sum(),axis=1)
        print(int_solution)
        print(i,'+++++++++++++++++++++++++++++++++++++++++')

    else:
    
#        tabu_list = []
#        cur_sol = int_solution
#        best_sol = cur_sol
        print(df_swap_.iloc[0:1, 0:15])

        
        print('A', full_str)
        print(len(df_iter['tablist'].loc[df_iter['tablist'] == str(full_str)]) == 0)

       
        if len(df_iter['tablist'].loc[df_iter['tablist'] == str(full_str)]) == 0:
            print('IF STATEMENT', i)

            df_iter = df_iter.append(df_swap_.loc[df_swap_['iteration'] == i].iloc[0, :])
            df_iter['tablist'] =  df_iter.apply(lambda row: row[0 : 15].sum(),axis=1)
            df_iter = df_iter.iloc[1:]
            print(int_solution)
            print(i,'+++++++++++++++++++++++++++++++++++++++++')   
#            tabu_list.append(int_solution)
#            df_iter = df_iter.append(int_solution)
#            print(tabu_list)
df_iter           
#%%
        
         len(df_iter['tablist'].loc[df_iter['tablist'] == str(full_str)]) 
        #%%
        df_iter.iloc[:, 0:15]
        #%%
    #        print(df_swap_.iloc[t , 15], flowcost(best_sol), df_swap_.iloc[t , 15] < flowcost(best_sol))
    #        print('--------------------------------------------------------------------------')
            if df_swap_['cost'].loc[df_swap_['iteration'] == i].iloc[0] < flowcost(best_sol):
                best_sol = df_swap_['cost'].loc[df_swap_['iteration'] == i].iloc[0]
               
                
                print(best_sol, flowcost(best_sol))
            
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            time.sleep(15)          
    
    
    
#    print(df_swap_['cost'].loc[df_swap_['iteration'] == i].iloc[0])
#    print('MARY ------------------------------------MARY')
    #print(print(df_swap_['cost'].loc[df_swap_['iteration'] == i]))

 #   df_iter = df_iter.append(df_swap_)


 
        #%%
        elif df_swap_.iloc[t , 15] < flowcost(best_sol):
            cur_sol = df_swap_.iloc[t , 15] 
            tabu_list.append(df_swap_.iloc[t , 0:-1].tolist())
            
            if len(tabu_list) > len_tabu:
                del tabu_list[0]
            
            best_sol = cur_sol
            
        

#%%
df_swap_.iloc[0 , 0:-1].tolist()
#%%
df_swap_.iloc[t , 15]


#%%
       #%%
        if sum_flowcost in dic_swap:          
            dic_swap[-sum_flowcost]= swap
            row_rep_not += 1
        else:
            dic_swap[sum_flowcost]= swap
        
        solution = p.vstack((solution,swap))
        
    

    import collections
    
    sorted_sol = collections.OrderedDict(sorted(dic_swap.items()))
    list_sorted_sol = list(sorted_sol.items())
    list_sorted_sol  
    
    #%%
    dic_swap[sum_flowcost]
    swap
    
    #%%
    sum_flowcost
    

#%%
    dictionary = {}
    dic_1 = Convert(dic_temp, dictionary)    
    dic_2 = Convert(dic_temp, dictionary)
    
    dic_1 =pd.DataFrame.from_dict(dic_1, orient='index')
    dic_2 =pd.DataFrame.from_dict(dic_2, orient='index')
    dic_1.columns = ['C1','C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11','C12', 'C13', 'C14', 'C15']
    dic_2.columns = ['C1','C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11','C12', 'C13', 'C14', 'C15']
    

    
    #%%
    dic_1.append(dic_2, ignore_index= True)
    
    #%%
    dic_1 = Convert(dic_temp, dictionary)
    
    #%%
    
    
    
 #%%   

                
      run=0  
        [3,4,5] = 41
        3,4 --> [4,3,5]= 30
        3,5 --> [5,4,3] =50
        4,5 --> [3,5,4]=60
        
        tabu_list = []
        run = 1
       best so far = [4,3,5] = 30
       4,3 = 
       4,5 =
       3,5 
        tabu_list= [(3,4)]
        
#%%     
    # check if the solution is in the tabu list 
    
    solutions = np.empty((0,15))

    def Convert(tup, di):
        di = dict(tup)
        return di 
#
    dictionary = {}
    list_sorted_sol_dic = Convert(list_sorted_sol, dictionary)
#    
    dic_values = []
    dic_keys = []
    tabu_list = []
    
    counter_key = 0
    for key, value in list_sorted_sol_dic.items():
#        print(key)
#        print(value)
        value = value.tolist() 
        dic_keys.append(key)
        dic_values.append(value)
    
    best_sol = int_solution
    index = -1
    for values_dic in dic_values:
        index +=1
        if values_dic not in tabu_list:
            tabu_list.append(values_dic)
        
            if len(tabu_list) >= len_tabu:
                del tabu_list[0]
            
        elif dic_keys[index] < flowcost(best_sol):
            best_sol = dic_values[index]
            best_sol_key = dic_keys[index]
            
     
        #%%
        df = pd.DataFrame().append(list_sorted_sol_dic, ignore_index=True)
             #%%        
        df =pd.DataFrame.from_dict(list_sorted_sol_dic, orient='index')
        df.columns = ['C1','C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11','C12', 'C13', 'C14', 'C15']
        df
#%%        
        best_sol=[]
        for key in sorted_sol:
            if key <= min(sorted_sol.keys()):
                best_sol = key      
    #%%
test_list = []
for i in range(10):
    print(i)
    test_list.append(i)
    
#%%
# find the objective function for each combination ; dictionary objective plus soution
#length of tabu list plus the fitness value 
print(min(dic_swap.keys()))

#%%
dic_swap[min(dic_swap.keys())]

#%% sort them based on the objective function 
for key in sorted(dic_swap):
    print (key, dic_swap[key])

#%%Use less than the whole neighborhood which is 105
'''neighbors_num = 20
for k in range(neighbors_num):
    random_num = random.randint(0, N-1)
    neighbors[k] = neighbors[random_num] '''




