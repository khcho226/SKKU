#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main() {
	int counter = 0;
	int counter2 = 0;
	int counter4 = 0;
	int counter5 = 0;
	int com;
	int num;
	char* arr[100000];
	char* arr2[100000];
	char* read2;

	FILE* txt = fopen("hw1_input.txt", "r");
	while (feof(txt) == 0) {
		read2 = (char*)malloc(sizeof(char) * 1500);
		fgets(read2, 1500, txt);
		if (counter > 3) {
			arr2[counter5] = read2;
			counter5++;
		}
		counter++;
	}
	fclose(txt);
	strcat(arr2[counter5 - 1], "\n");

	counter = 0;
	FILE* txt2 = fopen("hw1_input.txt", "r");
	while (feof(txt2) == 0) {
		counter++;
		char read[1500];
		fgets(read, 1500, txt2);

		if (counter == 1) {
			num = atoi(&read[0]);
		}

		if (counter == 3) {
			int length = sizeof(read) / sizeof(char);
			for (int i = 0; i < length; i++) {
				if (read[i] == ':') {
					counter2++;
				}
				if (read[i] == '*') {
					com = counter2 + 1;
					break;
				}
			}
		}

		if (counter > 4) {
			counter2 = 0;
			int counter3 = 0, i, start, end, length = sizeof(read) / sizeof(char);
			char* p_str, * p_sub;
			for (int i = 0; i < length; i++) {
				if (com - counter2 == 1) {
					if (counter3 == 0)
						start = i + 1;
					counter3 = 1;
				}
				if (com - counter2 == 0 || read[i] == '\0' || read[i] == '\n') {
					end = i - 1;
					break;
				}
				if (read[i] == ':') {
					counter2++;
				}
			}
			p_str = (char*)malloc(1500);
			p_sub = (char*)malloc(end - start + 2);
			p_str = read;
			for (i = 0; i < end - start + 1; i++) {
				*(p_sub + i) = *(p_str + start - 1 + i);
			}
			*(p_sub + i) = 0;
			arr[counter4] = p_sub;
			counter4++;
		}
	}
	fclose(txt2);

	char temp[30];
	char temp2[1500];
	for (int i = 0; i < num - 1; i++) {
		int min = i;
		for (int j = i + 1; j < num; j++) {
			if (strcmp(arr[j], arr[min]) < 0)
				min = j;
		}
		if (i != min) {
			strcpy(temp, arr[i]);
			strcpy(arr[i], arr[min]);
			strcpy(arr[min], temp);
			strcpy(temp2, arr2[i]);
			strcpy(arr2[i], arr2[min]);
			strcpy(arr2[min], temp2);
		}
	}

	FILE* fp = fopen("hw1_output.txt", "w");
	for (int i = 0; i < num; i++) {
		fprintf(fp, arr2[i]);
	}
	fclose(fp);
	printf("complete");

	return 0;
}