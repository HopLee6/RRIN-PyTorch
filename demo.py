import argparse
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from model import Net

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

def main():
    parser = argparse.ArgumentParser(description='PyTorch Video Frame Interpolation via Residue Refinement')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    transform = transforms.ToTensor()

    model = Net()
    state = torch.load('pretrained_model.pth.tar')
    model.load_state_dict(state,strict=True)
    model = model.cuda()
    model.eval()

    im1_path = 'data/im1.png'
    im2_path = 'data/im2.png'

    with torch.no_grad():

        img1 = Image.open(im1_path)
        img2 = Image.open(im2_path)

        img1 = transform(img1).unsqueeze(0).cuda()
        img2 = transform(img2).unsqueeze(0).cuda()
        
        if img1.size(1)==1:
            img1 = img1.expand(-1, 3,-1,-1)
            img2 = img2.expand(-1, 3,-1,-1)
            
        _,_,H,W = img1.size()
        H_,W_ = int(np.ceil(H/32)*32),int(np.ceil(W/32)*32)
        pader = torch.nn.ReplicationPad2d([0, W_-W , 0, H_-H])
        img1,img2 = pader(img1),pader(img2)
        
        output = model(img1, img2)
        output = output[0,:,0:H,0:W].squeeze(0).cpu()
        output = transforms.functional.to_pil_image(output)
        output.save('data/im_interp.png')

if __name__ == '__main__':
    main()
