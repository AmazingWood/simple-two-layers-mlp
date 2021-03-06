class buildConfig(object):
    def __init__(self,isDebug):
        self.incDirs=["/home/robin/installFromSource/boost_1_72_0","/home/robin/installFromSource/eigen-git-mirror"]
        self.linkDir=["/home/robin/installFromSource/boost_1_72_0/stage/lib"]
        self.linkOpt=["pthread" ,"m","dl"]
        self.CC="gcc-9"
        self.CXX="g++-9"
        self.CCFLAGS=['-std=c++17', '-Wall',"-fpermissive"]
        self.mklroot="/home/robin/intel/mkl"
        self.isDebug=int(isDebug)
        if(self.isDebug==1):
            self.preDifines=['-DDEBUG']
            self.CCFLAGS.append('-g')
        else:
            self.preDifines=['-DNDEBUG']
            self.CCFLAGS.append('-O3')


class MlpBuildConfig(buildConfig):
    
    def __init__(self,isDebug):
        
        buildConfig.__init__(self,isDebug)
        if(self.isDebug==1):
            self.targetName="mlpDebug"
        else:
            self.targetName="mlpRelease"
        self.incDirs.append(self.mklroot+"/include")
        self.linkDir.append(self.mklroot+"/lib/intel64")
        self.linkOpt.append(["mkl_intel_lp64","mkl_sequential" ,"mkl_core"])
        self.preDifines.append("EIGEN_USE_MKL_ALL")

#class LayersBuildConfig(buildConfig):
