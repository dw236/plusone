package plusone.utils;

import java.util.Random;
import java.util.HashSet;
import java.util.LinkedList;

public class Kmeans {
	
	private double[][] data;
	private int k;
	private String start;
	private String distance;
	private int rep;
	private int[] ID;
	private boolean[] empty;
	private double[][] bestCenters;
	private int[] bestID;
	private double minSumD;
	private double[][] centers;
	private double[] dToCenter;
	private int[] clusterSize;
	private double sumd;
	private int size;
	private int dim;
	
	private Random rand=new Random();
	public Kmeans(double[][] data,int k, int dim, String start, String distance, int rep)
	{
		this.data=data;
		this.start=start;
		this.distance=distance;
		this.k=k;
		this.rep=rep;
		this.dim = dim;
		this.size=data.length;
		ID=new int[size];
		centers=new double[k][dim];
		dToCenter=new double[size];
		clusterSize=new int[k];
		
	}
	
	public Kmeans(double[][] data, int k,int dim){
		this(data,k,dim,"cluster","cosine",50);
	}
	
	public double runKmeans(){
		minSumD=clustering();
		bestCenters=centers.clone();
		bestID=ID.clone();
		for (int i=0;i<rep-1;i++){
			double dd=clustering();
			if (dd<minSumD){
				minSumD=dd;
				bestCenters=centers.clone();
				bestID=ID.clone();
			}
		}
		return minSumD;
		
		
	}
	
	public double[][] getCenters(){
		return bestCenters;
	}
	
	public int[] getID(){
		return bestID;
	}
	public double getSumD(){
		return minSumD;
	}
	
	
	public double dotProduct(double[] a, double[] b){
		double result = 0.0;
		for (int i = 0; i < a.length; i ++){
			result += a[i] * b[i];
		}
		return result;
	}
	public double getDistance(double[] x,double[] y, String type){
		double result=0;
		switch (type) {
		case "euclidean":{
			for (int i=0;i<x.length;i++)
				result+=Math.pow((x[i]-y[i]),2.0);
			result=Math.sqrt(result);
			break;
		}
		case "cosine":{
			double xnorm=0;
			double ynorm=0;
			for (int i=0;i<x.length;i++){
				result+=x[i]*y[i];
				xnorm+=Math.pow(x[i],2);
				ynorm+=Math.pow(y[i],2);
			}
			result= 1-result/Math.sqrt(xnorm)/Math.sqrt(ynorm);
			break;
		}
		default: System.out.println("kmeans distance metric not supported");
				break;
		}
		return result;
		
	}
	
	
	
	public double clustering(){
		for (int i=0;i<size;i++) ID[i]=0;
		
		boolean converge=false;
		switch (start){
		case "sample": {
			HashSet<Integer> sample=new HashSet<Integer>(k);
			for (int i=0;i<k;i++){
				int j=rand.nextInt(size);
				while (sample.contains(j)){
					j=rand.nextInt(size);
				}
				centers[i]=data[j].clone();
			}
			break;
		}
		case "cluster":{
			double[][] subsample=new double[size/10][dim];
			HashSet<Integer> sample=new HashSet<Integer>(k);
			for (int i=0;i<subsample.length;i++){
				int j=rand.nextInt(size);
				while (sample.contains(j)){
					j=rand.nextInt(size);
				}
				subsample[i]=data[j];
			}
			
			Kmeans init = new Kmeans(subsample,k,dim,"sample","cosine",1);
			init.runKmeans();
			centers=init.getCenters();
			break;
		}
			
		default: System.out.println("kmeans starting method not supported");
				break;
		}
		while (!converge){
			
		converge = findCluster();
		
		if (!converge){
		for (int i=0;i<k;i++)
			if (clusterSize[i]==0){
				double max=dToCenter[0];
				int maxP=0;
				for (int j=1;j<size;j++)
					if (dToCenter[j]>max){
						max=dToCenter[j];
						maxP=j;
					}
				clusterSize[ID[maxP]]--;
				ID[maxP]=i;
				clusterSize[i]++;
				dToCenter[maxP]=0;
			}
		}
		
		if (!converge)
		findCenters();
				
		}
		return sumd;
			
	}
	public void findCenters(){
		
		for (int i=0;i<k;i++)
			for (int j=0;j<dim;j++)
				centers[i][j]=0;
		for (int i=0;i<size;i++){
			double[] p=data[i];
			for (int j=0;j<dim;j++)
				centers[ID[i]][j]+=p[j]/clusterSize[ID[i]];
		}
			
	}
	public boolean findCluster(){
		boolean noDiff=true;
		sumd=0;
		for (int i=0;i<k;i++) clusterSize[i]=0;
		for (int u=0;u<size;u++){
			double[] point=data[u];
			int c=0;
			double dist=getDistance(point,centers[0],this.distance);
			for (int i=1;i<k;i++){
				double d=getDistance(point,centers[i],this.distance);
				if (d<dist){
					dist=d;
					c=i;
				}
			}
			clusterSize[c]++;
			if (ID[u]!=c) noDiff=false;
			ID[u]=c;
			dToCenter[u]=dist;
			sumd+=dist;
		}
		return noDiff;
	}
}
