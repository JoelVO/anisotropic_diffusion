def diffusor(architecture, variance=None, function_type=None, niter=10, crop=256, option=None, gamma=None, degree=None, num_filters=None, depth=None, pad=8):
    """
    :param architecture: str. Available options: PeronaMalik, KAutomation, FoE, UNet.
    :param variance: int. Optional. Variance from the Gaussian noise the corrupted images are supposed to have.
    Trained models available when variance value is 15, 25 or 50. Default value: None
    :param function_type: str. Optional unless the architecture is FoE. If the selected arhitecture is
    FoE, it is compulsory and its available options are splines, decreasing, monomials and RothBlack. It is the family
    of functions where the diffusion function will be searched in.
    :param niter: int. Optional. Number of iterations for model has to reconstruct the images. Default value: 10.
    :param crop: int. Optional. Size of the images. Images are assumed to be squared. Default value: 256.
    :param option: 1 or 2. Optional. Only used when the selected architecture is KAutomation. If no value is provided,
    the returned model will be a pre-trained version for the KAutomation model using the non-exponential Perona-Malik
    diffusion function. Default value: None
    :param gamma: float. Optional. Time step size. If not provided, the used value will be that the models were trained
     with.
    :param degree: int. Optional. Kernel Size. Default value: None.
    :param num_filters: int. Optional. Number of kernels to be used when the FoE model is selected. Default value: None.
    :param depth: int. Optional. Number of convolutional blocks for the UNet whenever this architecture is selected.
    Default value: None.
    :return: If the selected architecture was PeronaMalik, it will return a Python function corresponding to the
    Perona-Malik model using the selected function for diffusing from options 1 (exponential) or 2 (non-exponential)
    alternatives.
    Otherwise, it will return a Keras model to reconstruct images using the desired architecture. In general, they will
    not be trained, but a trained version is also available. Take a look at the README for more information.
    """

    import pandas as pd

    if architecture == 'PeronaMalik':
        from pm_algorithm import anisodiff
        return anisodiff

    else:
        from architectures import get_nn
        import logging
        logging.getLogger('tensorflow').disabled = True

        if architecture == 'KAutomation':
            if option is None:
                model = get_nn(architecture=architecture, crop=crop, first=2, second=0, niter=niter, gamma=gamma)
                if variance is not None:
                    model.load_weights(f'./checkpoints/{variance}_2')
                return model
            else:
                return get_nn(architecture=architecture, crop=crop, first=option, second=0, niter=niter,gamma=gamma)

        elif architecture == 'FoE':
            if function_type is None:
                return None

            if (degree is not None) and (num_filters is not None):
                return get_nn(architecture=architecture, crop=crop, first=degree, second=num_filters, niter=niter, function_type=function_type, gamma=gamma)

            elif (degree is None) and (num_filters is None) and (variance is not None):
                df = pd.read_csv('./architecture_description.csv')
                search = (df.architecture == architecture) & (df.function_type == function_type) & (df.variance == variance)
                first, second = df[search].values[0, -2:]
                model = get_nn(architecture=architecture, crop=crop, first=first, second=second, niter=niter, function_type=function_type, gamma=gamma)
                model.load_weights(f'./checkpoints/{architecture}_{function_type}_{variance}_{first}_{second}')
                return model
            else:
                return None

        elif architecture == 'UNet':
            if function_type is not None:
                return None

            function_type = 'base'

            if (degree is not None) and (depth is not None):
                return get_nn(architecture=architecture, crop=crop, first=degree, second=depth, niter=niter, function_type=function_type, gamma=gamma)
            elif (degree is None) and (num_filters is None) and (variance is not None):
                df = pd.read_csv('./architecture_description.csv')
                search = (df.architecture == architecture) & (df.function_type == function_type) & (df.variance == variance)
                first, second = df[search].values[0, -2:]
                model = get_nn(architecture=architecture, crop=crop, first=first, second=second, niter=niter, function_type=function_type, gamma=gamma)
                model.load_weights(f'./checkpoints/{architecture}_{function_type}_{variance}_{first}_{second}')
                return model
            else:
                return None

        else:
            return None





if __name__ == '__main__':

    import sys
    import getopt

    architecture = None
    function_type = None
    variance = None
    niter = 10
    K, option, gamma = 50, None, 0.1
    images = None
    target_folder = None

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "h",
                               ["architecture=", "function_type=", "variance=", "niter=",
                                "K=", "option=", "gamma=", "images=", "target_folder="])

    description = """
        python anisotropic_diffusion.py --architecture --images --target_folder --variance --function_type --niter --K
         --option --gamma 

                    The only needed parameters are architecture and images.

                    architecture can be PeronaMalik, KAutomation, FoE or UNet.

                    images should be a folder with the images to reconstruct. If not provided, the model with the trained
                    weights will be returned.

                    When using the FoE architecture, the available function_type values are splines, decreasing, monomials
                    and RothBlack.   

                    If target_folder does not exist, it will be created. If it is not provided, a reconstructed folder
                     will be made inside the images folder. 
                    """

    for opt, arg in opts:
        if opt == '-h':
            print(description)
            sys.exit()
        elif opt == '--architecture':
            architecture = arg
        elif opt == '--function_type':
            function_type = arg
        elif opt == '--variance':
            variance = int(arg)
        elif opt == '--niter':
            niter = int(arg)
        elif opt == '--K':
            K = float(arg)
        elif opt == '--option':
            option = int(arg)
        elif opt == '--gamma':
            gamma = float(arg)
        elif opt == '--images':
            images = arg
        elif opt == '--target_folder':
            target_folder = arg

    import logging
    logging.getLogger('tensorflow').disabled = True

    if images is None:
        print("images need to be specified. Please give a folder with images to reconstruct")
        sys.exit()

    if (variance not in [15, 25, 50]) and (architecture != 'PeronaMalik'):
        message = """Variance value not provided which is necessary unless the PeronaMalik architecture is being used.
        Please select 15, 25 or 50 as possible values and try again or use the PeronaMalik architecture.
        """
        print(message)
        sys.exit()

    if architecture not in ['PeronaMalik', 'KAutomation', 'FoE', 'UNet']:
        message = """
        Requested architecture not valid.
        The valid options are: UNet, FoE and KAutomation.
        Please choose a valid option.
                   """
        print(message)
        sys.exit()

    if architecture == 'FoE':
        if function_type is None:
            function_type = 'splines'
        if function_type not in ['splines', 'decreasing', 'monomials', 'RothBlack']:
            message = """
            When the selected architecture is FoE, the valid function types are:
            splines, decreasing, monomials and RothBlack.
            Please choose a valid funtion type when using FoE as architecture.
            """
            print(message)
            sys.exit()

    if option not in [1, 2]:
        if architecture == 'PeronaMalik':
            message = """
            Invalid option value. It can be 1 or 2.
            """
            print(message)
            sys.exit()

    if (option != 2) and (architecture == 'KAutomation'):
        print("For this architecture, the only available trained model is for option 2. Changing option to 2")
        option = 2

    import pandas as pd
    from architectures import get_nn
    from pm_algorithm import anisodiff
    from glob import glob
    import cv2
    import numpy as np
    import os
    from tqdm import tqdm

    if target_folder is None:
        target_folder = f'{images}/reconstructed'
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    images = [log for log in glob(f'{images}/*') if not os.path.isdir(log)]
    images = np.stack(images)
    images_names = [im.split('/')[-1] for im in images]
    images = [cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2GRAY) for im in images]
    shapes = [im.shape for im in images]
    unique_shapes = np.unique(shapes, axis=0)

    if architecture == 'UNet':
        if function_type is not None:
            print('This architecture does not have options for function_type.')
        function_type = 'base'

    if architecture not in ['PeronaMalik', 'KAutomation']:
        df = pd.read_csv('./architecture_description.csv')
        if architecture == 'FoE':
            search = (df.architecture == architecture) & (df.function_type == function_type) & (df.variance == variance)
        if architecture == 'UNet':
            search = (df.architecture == architecture) & (df.variance == variance)
        first, second = df[search].values[0, -2:]
    else:
        first, second = 2, 0

    for crop in unique_shapes[:, 0]:
        print('Reconstructing images of size ', (crop, crop))
        images_size = [im for im in images if len(im) == crop]
        images_size_names = [images_names[i] for i in range(len(images)) if len(images[i]) == crop]

        if architecture in ['KAutomation', 'FoE', 'UNet']:
            model = get_nn(architecture, crop, first, second, niter=niter, function_type=function_type)
            if architecture != 'KAutomation':
                model.load_weights(f'./checkpoints/{architecture}_{function_type}_{variance}_{first}_{second}')
            else:
                model.load_weights(f'./checkpoints/{variance}_2')

            for i, image in tqdm(enumerate(images_size)):
                reconstructed = model(np.array([image])).numpy()[0]
                cv2.imwrite(f'{target_folder}/{images_size_names[i]}', reconstructed)

        if architecture == 'PeronaMalik':
            for i, image in tqdm(enumerate(images_size)):
                reconstructed = anisodiff(image, K=K, gamma=gamma, option=option, niter=niter)
                cv2.imwrite(f'{target_folder}/{images_size_names[i]}', reconstructed)

