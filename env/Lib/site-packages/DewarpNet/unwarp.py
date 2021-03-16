from .infer import *
from .infer import unwarp as base_unwarp

def unwarp(imgorg, wc_model_path=None, bm_model_path=None, compare=False):
    if wc_model_path == None:
        raise Exception('wc_model_path not specified: see https://github.com/cvlab-stonybrook/DewarpNet for more info')
    if bm_model_path == None:
        raise Exception('bm_model_path not specified: see https://github.com/cvlab-stonybrook/DewarpNet for more info')

    wc_model_file_name = os.path.split(wc_model_path)[1]
    wc_model_name = wc_model_file_name[:wc_model_file_name.find('_')]

    bm_model_file_name = os.path.split(bm_model_path)[1]
    bm_model_name = bm_model_file_name[:bm_model_file_name.find('_')]

    wc_n_classes = 3
    bm_n_classes = 2

    wc_img_size=(256,256)
    bm_img_size=(128,128)

    # Setup image
    imgorg = cv2.cvtColor(imgorg, cv2.COLOR_BGR2RGB)
    img = cv2.resize(imgorg, wc_img_size)
    img = img[:, :, ::-1]
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1) # NHWC -> NCHW
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Predict
    htan = nn.Hardtanh(0,1.0)
    wc_model = get_model(wc_model_name, wc_n_classes, in_channels=3)
    if DEVICE.type == 'cpu':
        wc_state = convert_state_dict(torch.load(wc_model_path, map_location='cpu')['model_state'])
    else:
        wc_state = convert_state_dict(torch.load(wc_model_path)['model_state'])
    wc_model.load_state_dict(wc_state)
    wc_model.eval()
    bm_model = get_model(bm_model_name, bm_n_classes, in_channels=3)
    if DEVICE.type == 'cpu':
        bm_state = convert_state_dict(torch.load(bm_model_path, map_location='cpu')['model_state'])
    else:
        bm_state = convert_state_dict(torch.load(bm_model_path)['model_state'])
    bm_model.load_state_dict(bm_state)
    bm_model.eval()

    if torch.cuda.is_available():
        wc_model.cuda()
        bm_model.cuda()
        images = Variable(img.cuda())
    else:
        images = Variable(img)

    with torch.no_grad():
        wc_outputs = wc_model(images)
        pred_wc = htan(wc_outputs)
        bm_input=F.interpolate(pred_wc, bm_img_size)
        outputs_bm = bm_model(bm_input)

    # call unwarp
    uwpred=base_unwarp(imgorg, outputs_bm)

    if compare:
        f1, axarr1 = plt.subplots(1, 2)
        axarr1[0].imshow(imgorg)
        axarr1[1].imshow(uwpred)
        plt.show()

    return uwpred

if __name__ == '__main__':

    
    wc_model_path = None
    bm_model_path = None

    self_dir = os.path.dirname(os.path.realpath(__file__))
    args_d = {
        'wc_model_path': os.path.join(self_dir, './eval/models/unetnc_doc3d.pkl'),
        'bm_model_path': os.path.join(self_dir, './eval/models/dnetccnl_doc3d.pkl'),
    }
    if wc_model_path == None:
        wc_model_path = args_d['wc_model_path']
    if bm_model_path == None:
        bm_model_path = args_d['bm_model_path']

    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    img_path_split = img_path.split('.')
    img_path_name = '.'.join(img_path_split[:-1])
    img_path_ext = img_path_split[-1]

    cv2.imshow("before", img)
    img = unwarp(img, wc_model_path, bm_model_path)
    cv2.imshow("after", img)
    # https://stackoverflow.com/a/58404775
    #img = cv2.convertScaleAbs(img, alpha=(255.0)) 
    #cv2.imwrite(img_path_name+"-uw."+img_path_ext, img)
    cv2.waitKey()