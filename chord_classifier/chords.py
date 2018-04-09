##
from soup import *

##
class Chords():
    """
    A class for generating simple synthetic chords for classification

    To get a random sample, use chords.rand()

    To get a batch for training, use chords.batch()

    """
    def __init__(self, n=2000, fs=4000, dur=0.5, base=440, sigma=1., phase_noise=1.):
        """
        Parameters
        ----------
        n : int
            number of examples per set (train/test/validate)
        fs : int
            sampling rate in Hz
        dur : int
            duration of tone produced in seconds
        base : int
            base frequency for calculations
        sigma : float
            std of random gaussian noise
        phase_noise : float
            scale of random phase shift
        """
        self.fs = fs
        self.t = np.arange(0, dur, 1/self.fs).astype(np.float32)
        self.clip_size = len(self.t)

        self.sigma = sigma
        self.phase_noise = phase_noise
        self.n_harmonics = 5
        self.harmonic_magnitudes = np.arange(0.1,3.1,0.1)

        self.abs_base = base
        self.roots = np.arange(-11,1) # 1 octaves for now
        self.qualities = [self.major,self.minor,self.dimin]#,self.aug,self.maj7,self.dom7,self.min7]
        self.magnitudes = np.arange(1,5)
        self.n_classes = len(self.qualities)

        self.train_data,self.train_labs = self.make_batch(n)
        self.val_data,self.val_labs = self.make_batch(n)
        self.test_data,self.test_labs = self.make_batch(n)
        self.i = 0

    def rand(self):
        # return a random chord
        root = np.random.choice(self.roots)
        qual = np.random.choice(self.qualities)
        chord = qual(root)

        print(root)
        print(qual)

        play(chord, rate=self.fs)
        return chord

    def from_base(self, n):
        # return the frequency of the note n half steps away from abs base
        return self.abs_base * (2**(1/12))**n

    def pure(self, f0):
        return np.sin(2*np.pi*f0*self.t + np.random.random()*self.phase_noise)

    def harmonics(self, f0):
        # generate random-magnitude harmonics from f0
        harm = np.zeros_like(self.t)
        for h in range(2,self.n_harmonics+2):
            harm += np.random.choice(self.harmonic_magnitudes) * self.pure(f0*h)
        return harm

    def onethree(self, root=0):
        return self.xad(root,0,4)

    def major(self, root=0):
        return self.xad(root,0,4,7)

    def minor(self, root=0):
        return self.xad(root,0,3,7)

    def aug(self, root=0):
        return self.xad(root,0,4,8)

    def dimin(self, root=0):
        return self.xad(root,0,3,6)
    
    def dom7(self, root=0):
        return self.xad(root,0,4,7,10)
    
    def maj7(self, root=0):
        return self.xad(root,0,4,7,11)
    
    def min7(self, root=0):
        return self.xad(root,0,3,7,10)

    def xad(self,root,*notes):
        # like triad, but for x notes
        ms = np.random.choice(self.magnitudes, size=len(notes))
        f0s = [self.from_base(root+n) for n in notes]
        x = np.zeros_like(self.t)
        for f0,m in zip(f0s,ms):
            x += m*self.pure(f0) + self.harmonics(f0)
        x += np.random.normal(0,self.sigma,size=x.shape)
        return x

    def make_batch(self, n):
        qual_idxs = np.random.choice(np.arange(len(self.qualities)), size=n)
        qual_arr = np.zeros([len(qual_idxs),len(self.qualities)])
        for i,qi in enumerate(qual_idxs):
            qual_arr[i,qi] = 1
        quals = np.array(self.qualities)[qual_idxs]
        roots = np.random.choice(self.roots, size=n)
        chords = np.array([q(r) for q,r in zip(quals,roots)])
        return chords, qual_arr

    def batch(self, n):
        x = np.take(self.train_data, np.arange(self.i,self.i+n), axis=0, mode='wrap')
        y = np.take(self.train_labs, np.arange(self.i,self.i+n), axis=0, mode='wrap')
        
        # increment counter
        self.i += n
        if self.i >= len(self.train_data):
            self.i -= len(self.train_data)

        return x,y

##
if __name__ == '__main__':

    c = Chords()
    #c.rand()
    b = c.batch(5)

##

