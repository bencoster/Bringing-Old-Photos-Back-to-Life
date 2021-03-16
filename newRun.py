import argparse

import cv2
import cv2 as cv
import skimage.io as io
import torchvision.transforms as transforms
from PIL import Image, ImageFile, ImageFilter
from openvino.inference_engine import IECore
from skimage import img_as_ubyte
from skimage.transform import SimilarityTransform
from skimage.transform import warp

import Face_Enhancement.data as data
from Face_Enhancement.models.pix2pix_model import Pix2PixModel
import Face_Enhancement.options.test_options as TestOptions
from Global.detection_util.util import *
from Global.models.mapping_model import Pix2PixHDModel_Mapping

import numpy as np

import torch.onnx
import onnx
import onnxruntime

ImageFile.LOAD_TRUNCATED_IMAGES = True


def checkNewModel(model, input, name, true_res):
    name = name + '.onnx'
    torch.onnx.export(model=model,  # model being run
                      args=input,  # model input (or a tuple for multiple inputs)
                      f=name,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,
                      do_constant_folding=True
                      )

    onnx_model = onnx.load(name)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(name)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    if type(input) is tuple:
        if len(ort_session.get_inputs()) > 1:
            ort_inputs = {ort_session.get_inputs()[i].name: to_numpy(input_i) for i, input_i in enumerate(input)}
        else:
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input[0])}
    else:
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(true_res), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def new_face_detector(image):
    plugin = IECore()

    device = 'CPU'

    FACE_DETECT_XML = "models/face-detection-adas-0001.xml"
    FACE_DETECT_BIN = "models/face-detection-adas-0001.bin"
    FACE_DETECT_INPUT_KEYS = 'data'
    FACE_DETECT_OUTPUT_KEYS = 'detection_out'
    net_face_detect = plugin.read_network(FACE_DETECT_XML, FACE_DETECT_BIN)
    # Load the Network using Plugin Device

    exec_face_detect = plugin.load_network(net_face_detect, device)

    # Obtain image_count, channels, height and width
    n_face_detect, c_face_detect, h_face_detect, w_face_detect = net_face_detect.input_info[
        FACE_DETECT_INPUT_KEYS].input_data.shape

    blob = cv.resize(image, (w_face_detect, h_face_detect))  # Resize width & height
    blob = blob.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    blob = blob.reshape((n_face_detect, c_face_detect, h_face_detect, w_face_detect))
    req_handle = exec_face_detect.start_async(
        request_id=0, inputs={FACE_DETECT_INPUT_KEYS: blob})

    req_handle.wait()
    res = req_handle.output_blobs[FACE_DETECT_OUTPUT_KEYS].buffer

    answer = []

    for detection in res[0][0]:  # TODO check if res[0][0] sorted by confidence
        confidence = float(detection[2])
        # Obtain Bounding box coordinate, +-10 just for padding
        xmin = int(detection[3] * image.shape[1] - 10)
        ymin = int(detection[4] * image.shape[0] - 10)
        xmax = int(detection[5] * image.shape[1] + 10)
        ymax = int(detection[6] * image.shape[0] + 10)
        face = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}
        if confidence > 0.9:
            answer.append(face)
    return answer


def new_unet_model(image):
    plugin = IECore()

    device = 'CPU'

    FACE_DETECT_XML = "models/Unet_model.xml"
    FACE_DETECT_BIN = "models/Unet_model.bin"
    FACE_DETECT_INPUT_KEYS = 'input.1'
    FACE_DETECT_OUTPUT_KEYS = '1085'
    net_face_detect = plugin.read_network(FACE_DETECT_XML, FACE_DETECT_BIN)
    # Load the Network using Plugin Device

    exec_face_detect = plugin.load_network(net_face_detect, device)

    # Obtain image_count, channels, height and width
    n_face_detect, c_face_detect, h_face_detect, w_face_detect = net_face_detect.input_info[
        FACE_DETECT_INPUT_KEYS].input_data.shape

    array = np.array(image)[0][0]

    w = array.shape[1]
    h = array.shape[0]

    blob = cv.resize(array, (w_face_detect, h_face_detect))  # Resize width & height
    blob = blob.reshape((n_face_detect, c_face_detect, h_face_detect, w_face_detect))
    req_handle = exec_face_detect.start_async(
        request_id=0, inputs={FACE_DETECT_INPUT_KEYS: blob})

    req_handle.wait()
    res = req_handle.output_blobs[FACE_DETECT_OUTPUT_KEYS].buffer
    # TODO IS IT WRONG TO CHANGE SIZE OF Scratches
    blob = cv.resize(res[0][0], (w, h))  # Resize width & height
    blob = blob.reshape((n_face_detect, c_face_detect, h, w))
    res = tv.transforms.ToTensor()(blob[0][0])
    res = torch.unsqueeze(res, 0)
    return res


def new_landmark_locator(image, current_face):
    plugin = IECore()

    device = 'CPU'

    FACE_DETECT_XML = "models/facial-landmarks-35-adas-0002.xml"
    FACE_DETECT_BIN = "models/facial-landmarks-35-adas-0002.bin"
    FACE_DETECT_INPUT_KEYS = 'data'
    FACE_DETECT_OUTPUT_KEYS = 'align_fc3'
    net_face_detect = plugin.read_network(FACE_DETECT_XML, FACE_DETECT_BIN)
    # Load the Network using Plugin Device

    exec_face_detect = plugin.load_network(net_face_detect, device)

    # Obtain image_count, channels, height and width
    n_face_detect, c_face_detect, h_face_detect, w_face_detect = net_face_detect.input_info[
        FACE_DETECT_INPUT_KEYS].input_data.shape
    face_xmin = current_face['xmin']
    face_ymin = current_face['ymin']
    face_xmax = current_face['xmax']
    face_ymax = current_face['ymax']

    face = image[face_ymin:face_ymax, face_xmin:face_xmax].copy()
    array = np.array(face)

    h, w = face.shape[0], face.shape[1]

    blob = cv.resize(array, (w_face_detect, h_face_detect))  # Resize width & height
    blob = blob.reshape((n_face_detect, c_face_detect, h_face_detect, w_face_detect))
    req_handle = exec_face_detect.start_async(
        request_id=0, inputs={FACE_DETECT_INPUT_KEYS: blob})

    req_handle.wait()
    res = req_handle.output_blobs[FACE_DETECT_OUTPUT_KEYS].buffer[0]
    for index in range(0, len(res), 2):
        res[index] = w - int(res[index] * w)# + face_xmin
        res[index + 1] = int(res[index + 1] * h)# + face_ymin

    for index in range(0, len(res), 2):
        face = cv2.circle(face, (res[index], res[index+1]), radius=1, color=(0, 0, 255), thickness=-1)
    return face


    x1, y1 = res[0], res[1]  # right corner of left eye
    x2, y2 = res[2], res[3]  # left corner of left eye
    x3, y3 = res[4], res[5]  # left corner of right eye
    x4, y4 = res[6], res[7]  # right corner of right eye

    x_nose, y_nose = int(res[8]), int(res[9])  # nose

    x_left_mouth, y_left_mouth = int(res[16]), int(res[17])
    x_right_mouth, y_right_mouth = int(res[18]), int(res[19])

    x_left_eye = int((x1 + x2) / 2)
    y_left_eye = int((y1 + y2) / 2)
    x_right_eye = int((x3 + x4) / 2)
    y_right_eye = int((y3 + y4) / 2)

    results = np.array(
        [
            [x_left_eye, y_left_eye],
            [x_right_eye, y_right_eye],
            [x_nose, y_nose],
            [x_left_mouth, y_left_mouth ],
            [x_right_mouth, y_right_mouth],
        ]
    )


def new_Pix2PixModel(data_i):
    plugin = IECore()

    device = 'CPU'

    FACE_DETECT_XML = "models/Pix2Pix.xml"
    FACE_DETECT_BIN = "models/Pix2Pix.bin"
    FACE_DETECT_INPUT_KEYS = 'degraded_image'
    FACE_DETECT_OUTPUT_KEYS = '818'
    net_face_detect = plugin.read_network(FACE_DETECT_XML, FACE_DETECT_BIN)
    # Load the Network using Plugin Device

    exec_face_detect = plugin.load_network(net_face_detect, device)

    # Obtain image_count, channels, height and width
    n_face_detect, c_face_detect, h_face_detect, w_face_detect = net_face_detect.input_info[
        FACE_DETECT_INPUT_KEYS].input_data.shape
    req_handle = exec_face_detect.start_async(
        request_id=0, inputs={FACE_DETECT_INPUT_KEYS: data_i})

    req_handle.wait()
    res = req_handle.output_blobs[FACE_DETECT_OUTPUT_KEYS].buffer

    return res


def new_Pix2PixModel_scratch(image, mask):
    plugin = IECore()

    device = 'CPU'

    FACE_DETECT_XML = "models/Pix2PixHDModel_Mapping_scratch.xml"
    FACE_DETECT_BIN = "models/Pix2PixHDModel_Mapping_scratch.bin"
    FACE_DETECT_INPUT_KEYS = 'mask.1'
    FACE_DETECT_OUTPUT_KEYS = '1334'
    net_face_detect = plugin.read_network(FACE_DETECT_XML, FACE_DETECT_BIN)
    # Load the Network using Plugin Device

    exec_face_detect = plugin.load_network(net_face_detect, device)
    array = np.array(mask)[0][0]

    # Obtain image_count, channels, height and width
    n_face_detect, c_face_detect, h_face_detect, w_face_detect = net_face_detect.input_info[
        FACE_DETECT_INPUT_KEYS].input_data.shape
    blob = cv.resize(array, (w_face_detect, h_face_detect))  # Resize width & height
    #blob = blob.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    blob = blob.reshape((n_face_detect, c_face_detect, h_face_detect, w_face_detect))
    req_handle = exec_face_detect.start_async(
        request_id=0, inputs={FACE_DETECT_INPUT_KEYS: blob[0]})

    req_handle.wait()
    res = req_handle.output_blobs[FACE_DETECT_OUTPUT_KEYS].buffer

    return res


def _standard_face_pts():
    pts = (np.array([196.0, 226.0, 316.0, 226.0, 256.0,
                     286.0, 220.0, 360.4, 292.0, 360.4], np.float32) / 256.0 - 1.0)
    return np.reshape(pts, (5, 2))


def data_transforms_global(img, full_size, method=Image.BICUBIC):
    if full_size == "full_size":
        ow, oh = img.size
        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == oh) and (w == ow):
            return img
        return img.resize((w, h), method)

    if full_size == "resize_256":
        return img.resize((parser.image_size, parser.image_size), method)

    if full_size == "scale_256":

        ow, oh = img.size
        pw, ph = ow, oh
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == ph) and (w == pw):
            return img
        return img.resize((w, h), method)


def data_transforms(img, method=Image.BILINEAR, scale=False):
    ow, oh = img.size
    pw, ph = ow, oh
    if scale:
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

    h = int(round(oh / 4) * 4)
    w = int(round(ow / 4) * 4)

    if (h == ph) and (w == pw):
        return img
    return img.resize((w, h), method)


def data_transforms_rgb_old(img):
    w, h = img.size
    A = img
    if w < 256 or h < 256:
        A = transforms.Scale(256, Image.BILINEAR)(img)
    return transforms.CenterCrop(256)(A)


def irregular_hole_synthesize(img, mask):
    img_np = np.array(img).astype("uint8")
    mask_np = np.array(mask).astype("uint8")
    mask_np = mask_np / 255
    img_new = img_np * (1 - mask_np) + mask_np * 255

    hole_img = Image.fromarray(img_new.astype("uint8")).convert("RGB")
    return hole_img

    # stage 4


def calculate_cdf(histogram):
    """
    This method calculates the cumulative distribution function
    :param array histogram: The values of the histogram
    :return: normalized_cdf: The normalized cumulative distribution function
    :rtype: array
    """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()

    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())
    return normalized_cdf


def calculate_lookup(src_cdf, ref_cdf):
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    """
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table


def match_histograms(src_image, ref_image):
    """
    This method matches the source image histogram to the
    reference signal
    :param image src_image: The original source image
    :param image  ref_image: The reference image
    :return: image_after_matching
    :rtype: image (array)
    """
    # Split the images into the different color channels
    # b means blue, g means green and r means red
    src_b, src_g, src_r = cv2.split(src_image)
    ref_b, ref_g, ref_r = cv2.split(ref_image)

    # Compute the b, g, and r histograms separately
    # The flatten() Numpy method returns a copy of the array c
    # collapsed into one dimension.
    src_hist_blue, bin_0 = np.histogram(src_b.flatten(), 256, [0, 256])
    src_hist_green, bin_1 = np.histogram(src_g.flatten(), 256, [0, 256])
    src_hist_red, bin_2 = np.histogram(src_r.flatten(), 256, [0, 256])
    ref_hist_blue, bin_3 = np.histogram(ref_b.flatten(), 256, [0, 256])
    ref_hist_green, bin_4 = np.histogram(ref_g.flatten(), 256, [0, 256])
    ref_hist_red, bin_5 = np.histogram(ref_r.flatten(), 256, [0, 256])

    # Compute the normalized cdf for the source and reference image
    src_cdf_blue = calculate_cdf(src_hist_blue)
    src_cdf_green = calculate_cdf(src_hist_green)
    src_cdf_red = calculate_cdf(src_hist_red)
    ref_cdf_blue = calculate_cdf(ref_hist_blue)
    ref_cdf_green = calculate_cdf(ref_hist_green)
    ref_cdf_red = calculate_cdf(ref_hist_red)

    # Make a separate lookup table for each color
    blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue)
    green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green)
    red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red)

    # Use the lookup function to transform the colors of the original
    # source image
    blue_after_transform = cv2.LUT(src_b, blue_lookup_table)
    green_after_transform = cv2.LUT(src_g, green_lookup_table)
    red_after_transform = cv2.LUT(src_r, red_lookup_table)

    # Put the image back together
    image_after_matching = cv2.merge([blue_after_transform, green_after_transform, red_after_transform])
    image_after_matching = cv2.convertScaleAbs(image_after_matching)

    return image_after_matching


# if compute_inverse_transformation_matrix set param inverse=True
def compute_transformation_matrix(img, landmark, normalize, target_face_scale=1.0, inverse=False):
    std_pts = _standard_face_pts()  # [-1,1]
    target_pts = (std_pts * target_face_scale + 1) / 2 * 256.0

    h, w, c = img.shape
    if normalize:
        landmark[:, 0] = landmark[:, 0] / h * 2 - 1.0
        landmark[:, 1] = landmark[:, 1] / w * 2 - 1.0

    affine = SimilarityTransform()
    if inverse:
        affine.estimate(landmark, target_pts)
    else:
        affine.estimate(target_pts, landmark)

    return affine


def blur_blending(im1, im2, mask):
    mask *= 255.0

    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    mask = Image.fromarray(mask.astype("uint8")).convert("L")
    im1 = Image.fromarray(im1.astype("uint8"))
    im2 = Image.fromarray(im2.astype("uint8"))

    mask_blur = mask.filter(ImageFilter.GaussianBlur(20))
    im = Image.composite(im1, im2, mask)

    im = Image.composite(im, im2, mask_blur)

    return np.array(im) / 255.0


def blur_blending_cv2(im1, im2, mask):
    mask *= 255.0

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=3)

    mask_blur = cv2.GaussianBlur(mask, (25, 25), 0)
    mask_blur /= 255.0

    im = im1 * mask_blur + (1 - mask_blur) * im2

    im /= 255.0
    im = np.clip(im, 0.0, 1.0)

    return im


def parameter_set(opt):
    # Default parameters
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.label_nc = 0
    opt.n_downsample_global = 3
    opt.mc = 64
    opt.k_size = 4
    opt.start_r = 1
    opt.mapping_n_block = 6
    opt.map_mc = 512
    opt.no_instance = True
    opt.checkpoints_dir = "Global/checkpoints/restoration"

    if opt.Quality_restore:
        opt.name = "mapping_quality"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_quality")
    if opt.Scratch_and_Quality_restore:
        opt.NL_res = True
        opt.use_SN = True
        opt.correlation_renormalize = True
        opt.NL_use_mask = True
        opt.NL_fusion_method = "combine"
        opt.non_local = "Setting_42"
        opt.name = "mapping_scratch"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_scratch")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="", help="Test images")
    parser.add_argument("--output_folder", type=str, help="Restored images, please use the absolute path")
    parser.add_argument("--checkpoint_name", type=str, default="Setting_9_epoch_100", help="choose which checkpoint")
    parser.add_argument("--with_scratch", action="store_true")
    opts = parser.parse_args()

    gpu = '-1'

    # resolve relative paths before changing directory
    opts.input_folder = os.path.abspath(opts.input_folder)
    opts.output_folder = os.path.abspath(opts.output_folder)
    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)

    main_environment = os.getcwd()

    # Stage 1: Overall Quality Improve
    print("Running Stage 1: Overall restoration")
    stage_1_input_dir = opts.input_folder
    stage_1_output_dir = os.path.join(opts.output_folder, "stage_1_restore_output")
    if not os.path.exists(stage_1_output_dir):
        os.makedirs(stage_1_output_dir)

    opt = argparse.ArgumentParser()
    opt.gpu_ids = [int(gpu)]
    opt.isTrain = False
    opt.resize_or_crop = 'scale_width'
    opt.input_nc = 3
    opt.output_nc = 3
    opt.ngf = 64
    opt.norm = 'instance'
    opt.spatio_size = 64
    opt.feat_dim = -1
    opt.use_segmentation_model = False
    opt.softmax_temperature = 1.0
    opt.use_self = False
    opt.cosin_similarity = False
    opt.mapping_net_dilation = 1
    opt.load_pretrain = ''
    opt.no_load_VAE = False
    opt.which_epoch = 'latest'
    opt.use_vae_which_epoch = 'latest'

    if not opts.with_scratch:

        opt.Scratch_and_Quality_restore = False
        opt.Quality_restore = True
        opt.test_input = stage_1_input_dir
        opt.outputs_dir = stage_1_output_dir
        opt.NL_use_mask = False
        opt.test_mode = 'Full'
        opt.non_local = ''
        parameter_set(opt)
        model = Pix2PixHDModel_Mapping()

        model.initialize(opt)
        model.eval()

        if not os.path.exists(f'{opt.outputs_dir}/input_image'):
            os.makedirs(f'{opt.outputs_dir}/input_image')
        if not os.path.exists(f'{opt.outputs_dir}/restored_image'):
            os.makedirs(f'{opt.outputs_dir}/restored_image')
        if not os.path.exists(f'{opt.outputs_dir}/origin'):
            os.makedirs(f'{opt.outputs_dir}/origin')

        input_loader = os.listdir(opt.test_input)
        dataset_size = len(input_loader)
        input_loader.sort()

        img_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        for i in range(dataset_size):

            input_name = input_loader[i]
            input_file = os.path.join(opt.test_input, input_name)
            if not os.path.isfile(input_file):
                print(f'Skipping non-file {input_name}')
                continue
            input = Image.open(input_file).convert("RGB")

            print(f'Now you are processing {input_name}')
            input = data_transforms(input, scale=False)
            origin = input
            input = img_transform(input)
            input = input.unsqueeze(0)
            mask = torch.zeros_like(input)

            try:
                generated = model.forward(input, mask)
                #checkNewModel(model, (input, mask), 'Pix2PixHDModel_Mapping', generated)
            except Exception as ex:
                print(f'Skip {input_name} due to an error:\n {str(ex)}')
                continue

            if input_name.endswith(".jpg"):
                input_name = f'{input_name[:-4]}.png'

            vutils.save_image(
                (input + 1.0) / 2.0,
                f'{opt.outputs_dir}/input_image/{input_name}',
                nrow=1,
                padding=0,
                normalize=True,
            )
            vutils.save_image(
                (generated.data.cpu() + 1.0) / 2.0,
                f'{opt.outputs_dir}/restored_image/{input_name}',
                nrow=1,
                padding=0,
                normalize=True,
            )

            origin.save(f'{opt.outputs_dir}/origin/{input_name}')

    else:
        mask_dir = os.path.join(stage_1_output_dir, "masks")
        new_input = os.path.join(mask_dir, "input")
        new_mask = os.path.join(mask_dir, "mask")

        print("initializing the dataloader")
        parser = argparse.ArgumentParser()

        parser.GPU = int(gpu)
        parser.test_path = stage_1_input_dir
        parser.output_dir = mask_dir
        parser.input_size = "scale_256"

        # dataloader and transformation
        print(f'directory of testing image: {parser.test_path}')
        imagelist = os.listdir(parser.test_path)
        imagelist.sort()
        total_iter = 0

        P_matrix = {}
        save_url = os.path.join(parser.output_dir)
        mkdir_if_not(save_url)

        input_dir = os.path.join(save_url, "input")
        output_dir = os.path.join(save_url, "mask")
        # blend_output_dir=os.path.join(save_url, 'blend_output')
        mkdir_if_not(input_dir)
        mkdir_if_not(output_dir)
        # mkdir_if_not(blend_output_dir)

        idx = 0

        for image_name in imagelist:

            idx += 1

            print("processing", image_name)

            results = []
            scratch_file = os.path.join(parser.test_path, image_name)
            if not os.path.isfile(scratch_file):
                print(f'Skipping non-file {image_name}')
                continue
            scratch_image = Image.open(scratch_file).convert("RGB")

            w, h = scratch_image.size

            transformed_image_PIL = data_transforms_global(scratch_image, parser.input_size)

            scratch_image = transformed_image_PIL.convert("L")
            scratch_image = tv.transforms.ToTensor()(scratch_image)

            scratch_image = tv.transforms.Normalize([0.5], [0.5])(scratch_image)

            scratch_image = torch.unsqueeze(scratch_image, 0)

            scratch_image = scratch_image.cpu()

            P = torch.sigmoid(new_unet_model(scratch_image))
            P = P.data.cpu()

            tv.utils.save_image(
                (P >= 0.4).float(),
                os.path.join(output_dir, f'{image_name[:-4]}.png'),
                nrow=1,
                padding=0,
                normalize=True,
            )
            transformed_image_PIL.save(os.path.join(input_dir, f'{image_name[:-4]}.png'))


        opt.Scratch_and_Quality_restore = True
        opt.test_input = new_input
        opt.test_mask = new_mask
        opt.outputs_dir = stage_1_output_dir
        opt.Quality_restore = False
        parameter_set(opt)
        model = Pix2PixHDModel_Mapping()

        model.initialize(opt)
        model.eval()

        if not os.path.exists(f'{opt.outputs_dir}/input_image'):
            os.makedirs(f'{opt.outputs_dir}/input_image')
        if not os.path.exists(f'{opt.outputs_dir}/restored_image'):
            os.makedirs(f'{opt.outputs_dir}/restored_image')
        if not os.path.exists(f'{opt.outputs_dir}/origin'):
            os.makedirs(f'{opt.outputs_dir}/origin')

        input_loader = os.listdir(opt.test_input)
        dataset_size = len(input_loader)
        input_loader.sort()

        if opt.test_mask:
            mask_loader = os.listdir(opt.test_mask)
            dataset_size = len(os.listdir(opt.test_mask))
            mask_loader.sort()

        img_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        mask_transform = transforms.ToTensor()

        for i in range(dataset_size):

            input_name = input_loader[i]
            input_file = os.path.join(opt.test_input, input_name)
            if not os.path.isfile(input_file):
                print(f'Skipping non-file {input_name}')
                continue
            input = Image.open(input_file).convert("RGB")

            print(f'Now you are processing {input_name}')

            if opt.NL_use_mask:
                mask_name = mask_loader[i]
                mask = Image.open(os.path.join(opt.test_mask, mask_name)).convert("RGB")
                origin = input
                input = irregular_hole_synthesize(input, mask)
                mask = mask_transform(mask)
                mask = mask[:1, :, :]  # Convert to single channel
                mask = mask.unsqueeze(0)
                input = img_transform(input)
                input = input.unsqueeze(0)
            else:
                if opt.test_mode == "Scale":
                    input = data_transforms(input, scale=True)
                if opt.test_mode == "Full":
                    input = data_transforms(input, scale=False)
                if opt.test_mode == "Crop":
                    input = data_transforms_rgb_old(input)
                origin = input
                input = img_transform(input)
                input = input.unsqueeze(0)
                mask = torch.zeros_like(input)
            # Necessary input

            try:
                generated = model.forward(input, mask)
                #new_generated = torch.from_numpy(new_Pix2PixModel_scratch(input, mask))
                #checkNewModel(model, (input, mask), 'Pix2PixHDModel_Mapping_scratch', generated)
            except Exception as ex:
                print(f'Skip {input_name} due to an error:\n {str(ex)}')
                continue

            if input_name.endswith(".jpg"):
                input_name = f'{input_name[:-4]}.png'

            vutils.save_image(
                (input + 1.0) / 2.0,
                f'{opt.outputs_dir}/input_image/{input_name}',
                nrow=1,
                padding=0,
                normalize=True,
            )
            vutils.save_image(
                (generated.data.cpu() + 1.0) / 2.0,
                f'{opt.outputs_dir}/restored_image/{input_name}',
                nrow=1,
                padding=0,
                normalize=True,
            )

            origin.save(f'{opt.outputs_dir}/origin/{input_name}')

    # Solve the case when there is no face in the old photo
    stage_1_results = os.path.join(stage_1_output_dir, "restored_image")
    stage_4_output_dir = os.path.join(opts.output_folder, "final_output")
    if not os.path.exists(stage_4_output_dir):
        os.makedirs(stage_4_output_dir)
    for x in os.listdir(stage_1_results):
        img_dir = os.path.join(stage_1_results, x)
        shutil.copy(img_dir, stage_4_output_dir)

    print("Finish Stage 1 ...\n")

    # Stage 2: Face Detection

    print("Running Stage 2: Face Detection")
    stage_2_input_dir = os.path.join(stage_1_output_dir, "restored_image")
    stage_2_output_dir = os.path.join(opts.output_folder, "stage_2_detection_output")
    if not os.path.exists(stage_2_output_dir):
        os.makedirs(stage_2_output_dir)

    url = stage_2_input_dir
    save_url = stage_2_output_dir

    # If the origin url is None, then we don't need to reid the origin image

    os.makedirs(url, exist_ok=True)
    os.makedirs(save_url, exist_ok=True)

    map_id = {}
    for x in os.listdir(url):
        img_url = os.path.join(url, x)
        pil_img = Image.open(img_url).convert("RGB")

        image = np.array(pil_img)

        faces = new_face_detector(image)

        if not faces:
            print(f'Warning: There is no face in {x}')
            continue
        else:
            for face_id, current_face in enumerate(faces):
                current_fl = new_landmark_locator(image, current_face)

                #affine = compute_transformation_matrix(image, current_fl, False, target_face_scale=1.3,
                #                                       inverse=False).params
                #aligned_face = warp(image, affine, output_shape=(256, 256, 3))
                img_name = f'{x[:-4]}_{face_id + 1}'
                io.imsave(os.path.join(save_url, f'{img_name}.png'), img_as_ubyte(current_fl))

    print("Finish Stage 2 ...\n")
    print("Please check stage 2 folder\n")
    exit()
    # Stage 3: Face Restore
    print("Running Stage 3: Face Enhancement")
    stage_3_input_mask = "./"
    stage_3_input_face = stage_2_output_dir
    stage_3_output_dir = os.path.join(opts.output_folder, "stage_3_face_output")
    if not os.path.exists(stage_3_output_dir):
        os.makedirs(stage_3_output_dir)

    opt = TestOptions.initialize(argparse.ArgumentParser())
    opt.old_face_folder = stage_3_input_face
    opt.old_face_label_folder = stage_3_input_mask
    opt.name = opts.checkpoint_name
    opt.gpu_ids = [int(gpu)]
    opt.results_dir = stage_3_output_dir
    dataloader = data.create_dataloader(opt)

    #model = Pix2PixModel(opt)
    #model.eval()

    single_save_url = os.path.join(opt.checkpoints_dir, opt.name, opt.results_dir, "each_img")

    if not os.path.exists(single_save_url):
        os.makedirs(single_save_url)

    for i, data_i in enumerate(dataloader):
        if i * opt.batchSize >= opt.how_many:
            break
        #generated_old = model(data_i, mode='inference')
        for j in range(len(data_i['image'])):
            generated = new_Pix2PixModel(data_i['image'][j].cpu())
            generated = torch.tensor(generated)
            img_path = data_i["path"][j]

            img_name = os.path.split(img_path)[-1]
            save_img_url = os.path.join(single_save_url, img_name)

            vutils.save_image((generated + 1) / 2, save_img_url)

    print("Finish Stage 3 ...\n")

    # Stage 4: Warp back
    print("Running Stage 4: Blending")
    stage_4_input_image_dir = os.path.join(stage_1_output_dir, "restored_image")
    stage_4_input_face_dir = os.path.join(stage_3_output_dir, "each_img")
    stage_4_output_dir = os.path.join(opts.output_folder, "final_output")
    if not os.path.exists(stage_4_output_dir):
        os.makedirs(stage_4_output_dir)

    origin_url = stage_4_input_image_dir
    replace_url = stage_4_input_face_dir
    save_url = stage_4_output_dir

    if not os.path.exists(save_url):
        os.makedirs(save_url)

    for x in os.listdir(origin_url):
        img_url = os.path.join(origin_url, x)
        pil_img = Image.open(img_url).convert("RGB")

        origin_width, origin_height = pil_img.size
        image = np.array(pil_img)

        faces = new_face_detector(image)

        if not faces:
            print(f'Warning: There is no face in {x}')
            continue

        blended = image
        for face_id, current_face in enumerate(faces):

            current_fl = new_landmark_locator(image, current_face)

            forward_mask = np.ones_like(image).astype("uint8")
            affine = compute_transformation_matrix(image, current_fl, False, target_face_scale=1.3, inverse=False)
            aligned_face = warp(image, affine, output_shape=(256, 256, 3), preserve_range=True)
            forward_mask = warp(
                forward_mask, affine, output_shape=(256, 256, 3), order=0, preserve_range=True
            )

            affine_inverse = affine.inverse
            cur_face = aligned_face
            if replace_url:
                face_name = f'{x[:-4]}_{face_id + 1}.png'
                cur_url = os.path.join(replace_url, face_name)
                restored_face = Image.open(cur_url).convert("RGB")
                restored_face = np.array(restored_face)
                cur_face = restored_face

            # Histogram Color matching
            A = cv2.cvtColor(aligned_face.astype("uint8"), cv2.COLOR_RGB2BGR)
            B = cv2.cvtColor(cur_face.astype("uint8"), cv2.COLOR_RGB2BGR)
            B = match_histograms(B, A)
            cur_face = cv2.cvtColor(B.astype("uint8"), cv2.COLOR_BGR2RGB)

            warped_back = warp(
                cur_face,
                affine_inverse,
                output_shape=(origin_height, origin_width, 3),
                order=3,
                preserve_range=True,
            )

            backward_mask = warp(
                forward_mask,
                affine_inverse,
                output_shape=(origin_height, origin_width, 3),
                order=0,
                preserve_range=True,
            )  # Nearest neighbour

            blended = blur_blending_cv2(warped_back, blended, backward_mask)
            blended *= 255.0

        io.imsave(os.path.join(save_url, x), img_as_ubyte(blended / 255.0))

    print("Finish Stage 4 ...\n")

    print("All the processing is done. Please check the results.")
