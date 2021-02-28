import os
import argparse

from torch.utils.data import DataLoader
from data_loader import GenerateDataset
from train import Sgg

parser_ssg = argparse.ArgumentParser(description='Argumentos para el entrenamiento.')
parser_ssg.add_argument('--path_train', type=str, default="g:/sgg/DIV2K_train_HR", help='Ruta con imagenes para el entrenamiento.')
parser_ssg.add_argument('--path_dev', type=str, default="g:/sgg/DIV2K_valid_HR", help='Ruta con imagenes para la evaluacion.')
parser_ssg.add_argument('--batch_size', type=int, default=32, help='batch size de entrenamiento.')
parser_ssg.add_argument('--epoch', type=int, default=200, help='Cantidad de epocas de entrenamiento.')
parser_ssg.add_argument('--sr', type=int, default=96, help='Tamaño a reescalar con sr.')
parser_ssg.add_argument('--encoder', type=str, default="vgg", help='Selecciona el encoder. vgg o mnv2')
parser_ssg.add_argument('--num_workers', type=int, default=4, help='Num workers para cargar el dataset en memoria.')
parser_ssg.add_argument('--shuffle', type=bool, default=True, help='Habilitar imagenes random.')
parser_ssg.add_argument('--fp16', type=bool, default=False, help='Habilitar modo AMP nvidia.')


def main():
    args = parser_ssg.parse_args()
    train_images = sorted([os.path.join(args.path_train, image) for image in os.listdir(args.path_train)])
    valid_images = sorted([os.path.join(args.path_dev, image) for image in os.listdir(args.path_dev)])

    train_data = DataLoader(GenerateDataset(train_images, h_hr=args.sr, w_hr=args.sr),
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers)
    dev_data = DataLoader(GenerateDataset(valid_images, h_hr=args.sr, w_hr=args.sr),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers)
    module = Sgg(args.encoder)
    module.train(train_data, dev_data, args.fp16, args.epoch)

if __name__ == '__main__':
    main()
