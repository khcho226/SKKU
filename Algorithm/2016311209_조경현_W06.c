#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main() {
	int counter = 0;
	int num;
	int len1, len2, len3, len4, len5;
	char arr[5][100];
	char ans0[200];
	char ans1[200];
	char ans2[500];

	FILE* txt = fopen("hw2_input.txt", "r");
	while (feof(txt) == 0) {
		counter++;
		char read[100];
		fgets(read, 100, txt);

		if (counter == 1)
			num = atoi(&read[0]);

		if (counter >= 3) {
			int length = sizeof(read) / sizeof(char);
			for (int i = 0; i < length; i++) {
				if (read[i] == '\0' || read[i] == '\n') {
					if (counter == 3) {
						len1 = i;
						for (int j = 0; j < length; j++)
							arr[0][j] = read[j];
					}
					else if (counter == 4) {
						len2 = i;
						for (int j = 0; j < length; j++)
							arr[1][j] = read[j];
					}
					else if (counter == 5) {
						len3 = i;
						for (int j = 0; j < length; j++)
							arr[2][j] = read[j];
					}
					else if (counter == 6) {
						len4 = i;
						for (int j = 0; j < length; j++)
							arr[3][j] = read[j];
					}
					else if (counter == 7) {
						len5 = i;
						for (int j = 0; j < length; j++)
							arr[4][j] = read[j];
					}
					break;
				}
			}
		}
	}

	int arr2[100][100];
	for (int i = 0; i < 100; i++) {
		arr2[i][0] = 0;
		arr2[0][i] = 0;
	}

	for (int i = 0; i < len1 + 1; i++) {
		for (int j = 0; j < len2 + 1; j++) {
			if (arr[0][i] == arr[1][j])
				arr2[i + 1][j + 1] = arr2[i][j] + 1;
			else {
				if (arr2[i + 1][j] >= arr2[i][j + 1])
					arr2[i + 1][j + 1] = arr2[i + 1][j];
				else
					arr2[i + 1][j + 1] = arr2[i][j + 1];
			}
		}
	}

	int x = 0;
	int y = 0;
	counter = 0;
	num = 0;
	while (1) {
		counter++;
		if (arr[0][len1 - 1] == arr[1][len2 - 1]) {
			ans0[x] = arr[0][len1 - 1];
			ans1[y] = arr[1][len2 - 1];
			len1--;
			len2--;
			x++;
			y++;
			num++;
		}
		else if (arr2[len1][len2] == arr2[len1 - 1][len2]) {
			ans0[x] = arr[0][len1 - 1];
			ans1[y] = '-';
			len1--;
			x++;
			y++;
			num++;
		}
		else if (arr2[len1][len2] == arr2[len1][len2 - 1]) {
			ans0[x] = '-';
			ans1[y] = arr[1][len2 - 1];
			len2--;
			x++;
			y++;
			num++;
		}
		if (len1 == 0 || len2 == 0)
			break;
	}

	int counter3 = 0;
	for (int i = num - 1; i >= 0; i--) {
		ans2[counter3] = ans0[i];
		counter3++;
	}
	ans2[counter3] = '\n';
	counter3++;

	for (int i = num - 1; i >= 0; i--) {
		ans2[counter3] = ans1[i];
		counter3++;
	}
	ans2[counter3] = '\n';
	counter3++;

	for (int i = num - 1; i >= 0; i--) {
		if (ans0[i] == ans1[i]) {
			ans2[counter3] = '*';
			counter3++;
		}
		else {
			ans2[counter3] = ' ';
			counter3++;
		}
	}

	FILE* fp = fopen("hw2_output.txt", "w");
	for (int i = 0; i < counter3; i++)
		fprintf(fp, "%c", ans2[i]);
	fclose(fp);
	printf("complete");

	return 0;
}