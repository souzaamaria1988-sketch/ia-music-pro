#!/usr/bin/env python3
"""
Treinamento integrado com Autoencoder
"""
import os,sys
from autoencoder import train_autoencoder

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=10000)
    parser.add_argument("--batch-size",type=int,default=64)
    args=parser.parse_args()
    
    print("🐧 Treinamento no Linux Ubuntu")
    train_autoencoder(epochs=args.epochs,batch_size=args.batch_size)
