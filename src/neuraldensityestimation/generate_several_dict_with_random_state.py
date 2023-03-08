def generate_several_dict_with_random_state(original_dict, number_of_experiments):
    
    return [dict(original_dict, **{"z_random_state": i}) for i in range(number_of_experiments)]
