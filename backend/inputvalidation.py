def input_validation(input_dict):
    def binary(input_dict, key):
        value = int(input_dict[key])
        return(value == 0 or value == 1)


    for key in input_dict:
        if(input_dict[key] == ""):
            error = f"Missing Input: {key}"
            print(error)
            return None, error

    #if passed in as floats, truncate all categorical inputs to ints. 
    input_dict['mental_health'] = int(input_dict['mental_health'])
    input_dict['physical_health'] = int(input_dict['physical_health'])
    input_dict['age'] = int(input_dict['age'])


    #check education level passed in as string format

    #based on this:
    '''
    1 = Never attended school or only kindergarten 
    2 = Grades 1 through 8 (Elementary) 
    3 = Grades 9 through 11 (Some high school) 
    4 = Grade 12 or GED (High school graduate) 
    5 = College 1 year to 3 years (Some college or technical school) 
    6 = College 4 years or more (College graduate)
    '''
    
    edu_level = input_dict['education'].lower()
    if(edu_level == 'kindergarten' or edu_level == 'never'):
        input_dict['education'] = 1
    elif(edu_level == 'elementary'):
        input_dict['education'] = 2
    elif(edu_level == 'some high school'):
        input_dict['education'] = 3
    elif(edu_level == 'high school graduate'):
        input_dict['education'] = 4
    elif(edu_level == 'some college'):
        input_dict['education'] = 5
    elif(edu_level == 'graduate'):
        input_dict['education'] = 6
    else:
        return None, "Error in: education"

    #set passed in age into appropriate category

    #Based on codebook values:
    '''
    1 Age 18 to 24
    2 Age 25 to 29
    3 Age 30 to 34
    4 Age 35 to 39
    5 Age 40 to 44
    6 Age 45 to 49
    7 Age 50 to 54
    8 Age 55 to 59
    9 Age 60 to 64
    10 Age 65 to 69
    11 Age 70 to 74
    12 Age 75 to 79
    13 Age 80 or older(99)
    age = input_dict['age']
    #check if possible to cast to an int, otherwise return None
    try:
        age = int(''.join(filter(str.isdigit, age)))
    except ValueError:
        return None
    '''
    age = input_dict['age']
    
    if 18 <= age <= 24:
        input_dict['age'] = 1
    elif 25 <= age <= 29:
        input_dict['age'] = 2
    elif 30 <= age <= 34:
        input_dict['age'] = 3
    elif 35 <= age <= 39:
        input_dict['age'] = 4
    elif 40 <= age <= 44:
        input_dict['age'] = 5
    elif 45 <= age <= 49:
        input_dict['age'] = 6
    elif 50 <= age <= 54:
        input_dict['age'] = 7
    elif 55 <= age <= 59:
        input_dict['age'] = 8
    elif 60 <= age <= 64:
        input_dict['age'] = 9
    elif 65 <= age <= 69:
        input_dict['age'] = 10
    elif 70 <= age <= 74:
        input_dict['age'] = 11
    elif 75 <= age <= 79:
        input_dict['age'] = 12
    elif 99 >= age >= 80:
        input_dict['age'] = 13
    else:
        return None, "Error in: age"

    #based on codebook:
    income = int(input_dict['income'])
    '''
    1 Less than $10,000
    2 Less than $15,000 ($10,000 to less than $15,000)
    3 Less than $20,000 ($15,000 to less than $20,000)
    4 Less than $25,000 ($20,000 to less than $25,000)
    5 Less than $35,000 ($25,000 to less than $35,000)
    6 Less than $50,000 ($35,000 to less than $50,000)
    7 Less than $75,000 ($50,000 to less than $75,000)
    8 $75,000 or more
    '''
    if income < 10000:
        input_dict['income'] = 1
    elif 10000 <= income < 15000:
        input_dict['income'] = 2
    elif 15000 <= income < 20000:
        input_dict['income'] = 3
    elif 20000 <= income < 25000:
        input_dict['income'] = 4
    elif 25000 <= income < 35000:
        input_dict['income'] = 5
    elif 35000 <= income < 50000:
        input_dict['income'] = 6
    elif 50000 <= income < 75000:
        input_dict['income'] = 7
    elif income >= 75000:
        input_dict['income'] = 8
    else:
        return None, "Error in: income"

    gen_health = input_dict['general_health']
    if gen_health == 'excellent':
        input_dict['general_health'] = 1
    elif gen_health == 'very good':
        input_dict['general_health'] = 2
    elif gen_health == 'good':
        input_dict['general_health'] = 3
    elif gen_health == 'fair':
        input_dict['general_health'] = 4
    elif gen_health == 'poor':
        input_dict['general_health'] = 5
    else:
        return None, "Error in: general_health"

    print("h")
    
    if (not binary(input_dict, 'high_bp')):
        return None, "Error in: high_bp"
        
    if (not binary(input_dict, 'high_chol')):
        return None, "Error in: high_chol"
        
    if (not binary(input_dict, 'chol_check')):
        return None, "Error in: chol_check"
        
    if(not (int(input_dict['bmi']) <= 100 and int(input_dict['bmi']) >= 0)):
        return None, "Error in: bmi"
        
    if(not binary(input_dict, 'smoker')):
        return None, "Error in: smoker"
        
    if(not binary(input_dict, 'stroke')):
        return None, "Error in: stroke"
        
    if(not binary(input_dict, 'heart_disease')):
        return None, "Error in: heart_disease"
        
    if(not binary(input_dict, 'physical_activity')):
        return None, "Error in: physical_activity"
        
    if(not binary(input_dict, 'fruits')):
        return None, "Error in: fruits"
        
    if(not binary(input_dict, 'veggies')):
        return None, "Error in: veggies"
        
    if(not binary(input_dict, 'heavy_drinker')):
        return None, "Error in: heavy_drinker"
        
    if(not binary(input_dict, 'healthcare')):
        return None, "Error in: healthcare"
        
    if(not binary(input_dict, 'no_doc_bc_cost')):
        return None, "Error in: no_doc_bc_cost"


    if(not (input_dict['general_health'] >=1 and input_dict['general_health']) <= 5):
        #checks for integer between 1 and 5 for the 5 categories. 
        return None, "Error in: general_health"
  
    if(not (input_dict['mental_health'] >=0 and input_dict['mental_health']) <= 30):
        return None, "Error in: mental_health"
 
    if(not (input_dict['physical_health'] >=0 and input_dict['physical_health'] <= 30)):
        return None, "Error in: physical_health"
        
    if(not binary(input_dict, 'diff_walk')):
        return None, "Error in: diff_walk"
        
    if(not binary(input_dict, 'sex')):
        return None, "Error in: sex"
        
    if(not (input_dict['age'] >= 0 and input_dict['age'] <= 13)):
        return None, "Error in: age"
    
    if(not (int(input_dict['education'] >= 1) and int(input_dict['education']) <= 6)):
        return None, "Error in: education" 
        
    if(not (input_dict['income'] >= 1 and input_dict['income'] <= 8)):
        return None, "Error in: income"

    return input_dict, ""
