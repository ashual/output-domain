from models.nn import *
import numpy as np
import itertools


class CoVAE32x32(nn.Module):
  def __init__(self, ch=32, input_dim_a=1, output_dim_a=1, input_dim_b=1, output_dim_b=1):
    super(CoVAE32x32, self).__init__()
    # Encoder layer #0
    self.g_en_conv0_a = LeakyReLUBNNSConv2d(input_dim_a, ch, kernel_size=5, stride=1, padding=2)
    self.g_en_conv0_b = LeakyReLUBNNSConv2d(input_dim_b, ch, kernel_size=5, stride=1, padding=2)
    self.g_en_conv1 = LeakyReLUBNNSConv2d(ch * 1, ch * 2, kernel_size=5, stride=2, padding=2)
    self.g_en_conv2 = LeakyReLUBNNSConv2d(ch * 2, ch * 4, kernel_size=8, stride=1, padding=0)
    self.g_en_conv3 = LeakyReLUBNNSConv2d(ch * 4, ch * 8, kernel_size=1, stride=1, padding=0)
    # Latent layer
    self.g_vae = GaussianVAE2D(ch * 8, ch * 8, kernel_size=1, stride=1)
    # Decoder layer #0
    self.g_de_conv0 = LeakyReLUBNNSConvTranspose2d(ch * 8, ch * 8, kernel_size=4, stride=2, padding=0)
    self.g_de_conv1 = LeakyReLUBNNSConvTranspose2d(ch * 8, ch * 4, kernel_size=4, stride=2, padding=1)
    self.g_de_conv2 = LeakyReLUBNNSConvTranspose2d(ch * 4, ch * 2, kernel_size=4, stride=2, padding=1)
    self.g_de_conv3_a = LeakyReLUBNNSConvTranspose2d(ch * 2, ch * 1, kernel_size=4, stride=2, padding=1)
    self.g_de_conv3_b = LeakyReLUBNNSConvTranspose2d(ch * 2, ch * 1, kernel_size=4, stride=2, padding=1)
    # Decoder layer #4
    self.de_conv4_a = nn.ConvTranspose2d(ch * 1, output_dim_a, kernel_size=1, stride=1, padding=0)
    self.de_conv4_b = nn.ConvTranspose2d(ch * 1, output_dim_b, kernel_size=1, stride=1, padding=0)
    self.de_tanh4_a = nn.Tanh()
    self.de_tanh4_b = nn.Tanh()
    xy = self._create_xy_image()
    self.xy = xy.unsqueeze(0).expand(128, xy.size(0), xy.size(1), xy.size(2))

  def _create_xy_image(self, width=32):
    coordinates = list(itertools.product(range(width), range(width)))
    arr = (np.reshape(np.asarray(coordinates), newshape=[width, width, 2]) - width/2 ) / (width/2)
    new_map = np.transpose(np.float32(arr), [2, 0, 1])
    xy = Variable(torch.from_numpy(new_map), requires_grad=False)
    return xy

  def forward(self, x_a, x_b):
    # x_a = torch.cat((x_a, self.xy), 1)  # Create input image to the generator a
    # x_b = torch.cat((x_b, self.xy), 1)  # Create input image to the generator b
    en_h0_a = self.g_en_conv0_a(x_a)
    en_h0_b = self.g_en_conv0_b(x_b)
    en_h0 = torch.cat((en_h0_a, en_h0_b), 0)
    en_h1 = self.g_en_conv1(en_h0)
    en_h2 = self.g_en_conv2(en_h1)
    en_h3 = self.g_en_conv3(en_h2)
    z, mu, sd = self.g_vae.sample(en_h3)
    de_h0 = self.g_de_conv0(z)
    de_h1 = self.g_de_conv1(de_h0)
    de_h2 = self.g_de_conv2(de_h1)
    de_h3_a = self.g_de_conv3_a(de_h2)
    de_h3_b = self.g_de_conv3_b(de_h2)
    de_h4_a = self.de_tanh4_a(self.de_conv4_a(de_h3_a))
    de_h4_b = self.de_tanh4_b(self.de_conv4_b(de_h3_b))
    x_aa, x_ba = torch.split(de_h4_a, x_a.size(0), dim=0)
    x_ab, x_bb = torch.split(de_h4_b, x_a.size(0), dim=0)
    codes = (mu, sd)
    return x_aa, x_ba, x_ab, x_bb, [codes]
