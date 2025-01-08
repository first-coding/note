```
Class SRCNN(nn.Module):
	def __init__(self, scale_factor=3): 
	super(SRCNN, self).__init__() 
	self.scale_factor = scale_factor 
	# 定义三个主要的卷积层 
	self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4) 
	self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0) 
	self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2) 
	
	def forward(self, x):
		x = torch.relu(self.conv1(x)) 
		x = torch.relu(self.conv2(x)) 
		x = self.conv3(x) 
		return x
```

^45a59d

```
生成器：
class Generator(nn.Module): 
	def __init__(self, scale_factor=4):
		super(Generator, self).__init__()
		self.scale_factor = scale_factor
		self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4) 
		self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) 
		self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4) 
			
	def forward(self, x): 
		x = F.relu(self.conv1(x)) 
		x = F.relu(self.conv2(x)) 
		x = self.conv3(x) 
		return x

判别器：
class Discriminator(nn.Module): 
	def __init__(self): 
		super(Discriminator, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1) 
		self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) 
		self.fc = nn.Linear(128*16*16, 1) # 假设输入尺寸为64x64
			 
	def forward(self, x): 
		x = F.relu(self.conv1(x)) 
		x = F.relu(self.conv2(x)) 
		x = x.view(x.size(0), -1) # 展平 
		x = torch.sigmoid(self.fc(x)) 
		return x
```

^c6e0f6

```
生成器：
class EnhancedGenerator(nn.Module): 
	def __init__(self, scale_factor=4):
		super(EnhancedGenerator, self).__init__()
		self.scale_factor = scale_factor 
		self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4) 
		self.res_blocks = self._make_res_blocks(64, 64, 16) # 使用16个残差块 
		self.conv2 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4) 
			
	def _make_res_blocks(self, in_channels, out_channels, num_blocks): 
		layers = [] 
		for _ in range(num_blocks):
			layers.append(self._residual_block(in_channels, out_channels)) 
		return nn.Sequential(*layers) 
	    
	def _residual_block(self, in_channels, out_channels):
		return nn.Sequential( 
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1)
		) 
		
	def forward(self, x): 
		x = F.relu(self.conv1(x)) 
		x = self.res_blocks(x) 
		x = self.conv2(x) 
		return x
```

^69399d

