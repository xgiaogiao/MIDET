from .midet import MIDET
def model_generator(opt, device="cuda"):
    method = opt
   
    if 'midet' in method:
        num_iterations = int(method.split('_')[-1])
        model = MIDET(in_c=28, n_feat=28,nums_stages=num_iterations-1).to(device)
    else:
        print(f'opt.Method {opt.method} is not defined !!!!')
    
    return model