#!/usr/bin/env python

from __future__ import print_function
import hashlib
import os
import sys
import tarfile
import requests

from urllib.request import urlopen


class Model:
    MB = 1024*1024
    BUFSIZE = 10*MB

    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.url = kwargs.pop('url', None)
        self.downloader = kwargs.pop('downloader', None)
        self.filename = kwargs.pop('filename')
        self.sha = kwargs.pop('sha', None)
        self.archive = kwargs.pop('archive', None)
        self.member = kwargs.pop('member', None)

    def __str__(self):
        return 'Model <{}>'.format(self.name)

    def printRequest(self, r):
        def getMB(r):
            d = dict(r.info())
            for c in ['content-length', 'Content-Length']:
                if c in d:
                    return int(d[c]) / self.MB
            return '<unknown>'
        print('  {} {} [{} Mb]'.format(r.getcode(), r.msg, getMB(r)))

    def verify(self):
        if not self.sha:
            return False
        print('  expect {}'.format(self.sha))
        sha = hashlib.sha1()
        try:
            with open(self.filename, 'rb') as f:
                while True:
                    buf = f.read(self.BUFSIZE)
                    if not buf:
                        break
                    sha.update(buf)
            print('  actual {}'.format(sha.hexdigest()))
            return self.sha == sha.hexdigest()
        except Exception as e:
            print('  catch {}'.format(e))

    def get(self):
        if self.verify():
            print('  hash match - skipping')
            return True

        basedir = os.path.dirname(self.filename)
        if basedir and not os.path.exists(basedir):
            print('  creating directory: ' + basedir)
            os.makedirs(basedir, exist_ok=True)

        if self.archive or self.member:
            assert(self.archive and self.member)
            print('  hash check failed - extracting')
            print('  get {}'.format(self.member))
            self.extract()
        elif self.url:
            print('  hash check failed - downloading')
            print('  get {}'.format(self.url))
            self.download()
        else:
            assert self.downloader
            print('  hash check failed - downloading')
            sz = self.downloader(self.filename)
            print('  size = %.2f Mb' % (sz / (1024.0 * 1024)))

        print(' done')
        print(' file {}'.format(self.filename))
        return self.verify()

    def download(self):
        try:
            r = urlopen(self.url, timeout=60)
            self.printRequest(r)
            self.save(r)
        except Exception as e:
            print('  catch {}'.format(e))

    def extract(self):
        try:
            with tarfile.open(self.archive) as f:
                assert self.member in f.getnames()
                self.save(f.extractfile(self.member))
        except Exception as e:
            print('  catch {}'.format(e))

    def save(self, r):
        with open(self.filename, 'wb') as f:
            print('  progress ', end='')
            sys.stdout.flush()
            while True:
                buf = r.read(self.BUFSIZE)
                if not buf:
                    break
                f.write(buf)
                print('>', end='')
                sys.stdout.flush()


def GDrive(gid):
    def download_gdrive(dst):
        session = requests.Session()  # re-use cookies

        URL = "https://docs.google.com/uc?export=download"
        response = session.get(URL, params = { 'id' : gid }, stream = True)

        def get_confirm_token(response):  # in case of large files
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value
            return None
        token = get_confirm_token(response)

        if token:
            params = { 'id' : gid, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)

        BUFSIZE = 1024 * 1024
        PROGRESS_SIZE = 10 * 1024 * 1024

        sz = 0
        progress_sz = PROGRESS_SIZE
        with open(dst, "wb") as f:
            for chunk in response.iter_content(BUFSIZE):
                if not chunk:
                    continue  # keep-alive

                f.write(chunk)
                sz += len(chunk)
                if sz >= progress_sz:
                    progress_sz += PROGRESS_SIZE
                    print('>', end='')
                    sys.stdout.flush()
        print('')
        return sz
    return download_gdrive


models = [

    Model(
        name='Unet_model.xml',
        url='https://drive.google.com/uc?export=dowload&id=1oUp4x_Kr9Mt_N5SAh3Ypy-09qri2kbzj',
        sha='d9f76173f6ebf97093abccc5e4f62553e0ad97e6',
        filename='models/Unet_model.xml'),
    Model(
        name='Unet_model.bin',
        downloader=GDrive('1cOGW879DF505EodUpmGtzoVcFRADDRuP'),
        sha='79e9abdfc89eed5095b369a6ac9e5594e3fc4f3c',
        filename='models/Unet_model.bin'),
    Model(
        name='face-detection-adas-0001.bin',
        url='https://drive.google.com/uc?export=dowload&id=1M_Vb0iG2NIs4DrwqWVavEMahn5xRgfLK',
        sha='43dd6c8512bc578671733b03f2ea80d292b93d92',
        filename='models/face-detection-adas-0001.bin'),
    Model(
        name='face-detection-adas-0001.xml',
        url='https://drive.google.com/uc?export=dowload&id=16syJHdW_1m_UO6b2vsApEm-LkMzLsvo4',
        sha='4ae7758827eea7b7f4e37fffeec3c793042f8dcd',
        filename='models/face-detection-adas-0001.xml'),
    Model(
        name='facial-landmarks-35-adas-0002.xml',
        url='https://drive.google.com/uc?export=dowload&id=1Dl0gP2bX15Efm-RjNB6FxF7ryiEfxwfl',
        sha='a0c3ed72898c77c76f68a274ab5ee7766093e5f4',
        filename='models/facial-landmarks-35-adas-0002.xml'),
    Model(
        name='facial-landmarks-35-adas-0002.bin',
        url='https://drive.google.com/uc?export=dowload&id=1SLSVGzV7QTCaHBWqTsyzb2sDWrzoRLeg',
        sha='aaf693b07978889ad229459c29ed89ca1630e711',
        filename='models/facial-landmarks-35-adas-0002.bin'),
    Model(
        name='Pix2Pix.xml',
        url='https://drive.google.com/uc?export=dowload&id=1Cz4p6yaf1uck4aYViYYJ54NTnjR_BQ8-',
        sha='6c0d4bd39bfb602f58791dc6c247ea9adf54e1fb',
        filename='models/Pix2Pix.xml'),
    Model(
        name='Pix2Pix.bin',
        downloader=GDrive('1Najox7cEE7x26VvOLLnH3q6ArL16KT8t'),
        sha='9dc129e003b2a120820a44bbe67184757e0da52a',
        filename='models/Pix2Pix.bin'),
    Model(
        name='Pix2PixHDModel_Mapping_No_Scratch.xml',
        url='https://drive.google.com/uc?export=dowload&id=1taB4k0Hbv41qFa_LTj3a7NCIIyfKbIDc',
        sha='fd390507e28ab40492fac083cef084b742872790',
        filename='models/Pix2PixHDModel_Mapping_No_Scratch.xml'),
    Model(
        name='Pix2PixHDModel_Mapping_No_Scratch.bin',
        downloader=GDrive('1tFnVt1aROd58Tb3qdawUyN1cwE74R5i3'),
        sha='4827842923e65638403ff4f9ecb409d634b46467',
        filename='models/Pix2PixHDModel_Mapping_No_Scratch.bin'),
    Model(
        name='Pix2PixHDModel_Mapping_scratch.xml',
        url='https://drive.google.com/uc?export=dowload&id=1UYAm1lIC--7YMtJo7_rPMvSCG0l16r6T',
        sha='563d3185d65083d17946f4bfbbba0808596b537c',
        filename='models/Pix2PixHDModel_Mapping_scratch.xml'),
    Model(
        name='Pix2PixHDModel_Mapping_scratch.bin',
        downloader=GDrive('1e3oRHRC16rNA9u4bMm8sAA9IkLtBZuo6'),
        sha='56ad193120bbff418d373736a20b0ea3984a6e05',
        filename='models/Pix2PixHDModel_Mapping_scratch.bin'),


]

# Note: models will be downloaded to current working directory
#       expected working directory is <testdata>/dnn
if __name__ == '__main__':

    selected_model_name = None
    if len(sys.argv) > 1:
        selected_model_name = sys.argv[1]
        print('Model: ' + selected_model_name)

    failedModels = []
    for m in models:
        print(m)
        if selected_model_name is not None and not m.name.startswith(selected_model_name):
            continue
        if not m.get():
            failedModels.append(m.filename)

    if failedModels:
        print("Following models have not been downloaded:")
        for f in failedModels:
            print("* {}".format(f))
        exit(15)