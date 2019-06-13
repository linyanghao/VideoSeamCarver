from video import *

class Indicator():

    def isInTarget(x, y):
        return False
    
class SquareIndicator(Indicator):
    def __init__(self, ux, uy, bx, by):
        if ux >= bx or uy <= by:
            raise KeyError("Wrong rectangle!")
        self.ux = ux
        self.uy = uy
        self.bx = bx
        self.by = by


    def isInTarget(self, target_x, target_y):
        return self.ux < target_x and target_x < self.bx \
            and self.uy > target_y and target_y > self.by

class ContentRemover(VideoSeamCarver):

    def __init__(self, img, indicator, mode='vertical'):
        self.indicator = indicator
        VideoSeamCarver.__init__(self, img, mode)

    def _energy(self, t, i, j): 
        # print("here")
        old_e = super()._energy(t, i, j)
        return old_e if not self.indicator.isInTarget(i, j) else -old_e

if __name__ == '__main__':
    IMAGE_AS_VIDEO, SMALL_DATA = True, False

    REMOVE_SEAM_TEST  = True
    AUGMENT_SEAM_TEST = False
    XY_SCALE_TEST     = False

    assert XY_SCALE_TEST + REMOVE_SEAM_TEST + AUGMENT_SEAM_TEST == 1, "Wrong setting!"

    REMOVE_SEAMS_COUNT  = 40
    AUGMENT_SEAMS_COUNT = 40
    X_SEAMS_COUNT       = 20
    Y_SEAMS_COUNT       = -13
    
    if IMAGE_AS_VIDEO:
        img   = Image.open('2.png')
        img   = img.convert('RGB')
        img   = np.array(img)
        video = np.reshape(img, [1, *img.shape])
        print(video.shape)
    else:
        video = imageio.get_reader('golf.mov', 'ffmpeg')
        frames = []
        for i, frame in enumerate(video):
            frames.append(frame)
        video = np.array(frames)

    if SMALL_DATA:
        video = video[:10, ...]

    carver = ContentRemover(video,  \
        indicator=SquareIndicator(100, 150, 150, 100))
    #carver.Draw()

    if REMOVE_SEAM_TEST: # 测试减少图片宽度的功能
        REMOVE_SEAMS_COUNT = 20
        print("=========== REMOVE_SEAM_TEST ==============\n")
        res = carver.shrink_hor(REMOVE_SEAMS_COUNT, save_hist=True)
        
        assert res.shape[2] == video.shape[2] - REMOVE_SEAMS_COUNT, "{} is not {}".format(
           res.shape[2], video.shape[2] - REMOVE_SEAMS_COUNT
        )

        # make an animation
        images = []
        for i in range(0,REMOVE_SEAMS_COUNT):
            filename = OUT_FOLDER + "/{}.gif".format(i)
            images.append(imageio.imread(filename))
        imageio.mimsave(OUT_FOLDER + "/movie.gif", images)

        carver.augment_hor(REMOVE_SEAMS_COUNT, save_hist=True)



