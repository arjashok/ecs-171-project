#helper function, corrects/validates inputs passed in
def input_validation(input_dict):
    def binary(input_dict, key):
        return(input[key] == 0 or input[key] == 1)

    #if passed in as floats, truncate all categorical inputs to ints. 
    input_dict['general_health'] = int(input_dict['general_health'])
    input_dict['mental_health'] = int(input_dict['mental_health'])
    input_dict['physical_health'] = int(input_dict['physical_health'])
    input_dict['education'] = int(input_dict['education'])
    input_dict['age'] = int(input_dict['age'])
    input_dict['income'] = int(input_dict['income'])


    
    if (not binary(input_dict, 'high_bp')):
        return None
        
    if (not binary(input_dict, 'high_chol')):
        return None
        
    if (not binary(input_dict, 'chol_check')):
        return None
        
    if(not (input_dict['bmi'] <= 100 and input_dict['bmi'] >= 0)):
        return None
        
    if(not binary(input_dict, 'smoker')):
        return None
        
    if(not binary(input_dict, 'stroke')):
        return None
        
    if(not binary(input_dict, 'heart_disease')):
        return None
        
    if(not binary(input_dict, 'physical_activity')):
        return None
        
    if(not binary(input_dict, 'fruits')):
        return None
        
    if(not binary(input_dict, 'veggies')):
        return None
        
    if(not binary(input_dict, 'heavy_drinker')):
        return None
        
    if(not binary(input_dict, 'healthcare')):
        return None
        
    if(not binary(input_dict, 'no_doc_bc_cost')):
        return None


    if(not (input_dict['general_health'] >=1 and input_dict['general_health'] <= 5)):
        #checks for integer between 1 and 5 for the 5 categories. 
        return None
  
    if(not (input_dict['mental_health'] >=0 and input_dict['mental_health'] <= 30)):
        return None
 
    if(not (input_dict['physical_health'] >=0 and input_dict['physical_health'] <= 30)):
        return None
        
    if(not binary(input_dict, 'diff_walk')):
        return None
        
    if(not binary(input_dict, 'sex')):
        return None
        
    if(not (input_dict['age'] >= 0 and input_dict['age'] <= 13)):
        return None
    
    if(not (input_dict['education'] >= 1 and input_dict['education'] <= 6)):
        return None   
        
    if(not (input_dict['income'] >= 1 and input_dict['income'] <= 8)):
        return None

    return input_dict
    

    
        
        
    

    
    