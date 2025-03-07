import os
import csv
import tqdm

class FileIO:
    def __init__(self) -> None:
        pass

    def Read(self, filePath, lineDelimiter: str = '') -> list:
        """
        File reader

        Parameters
        ----------
        filePath: str
            File path. Include extension.
        lineDelimiter: (Optional) str
            The delimiter for each line.
        
        Returns
        ----------
        data: list
            Read data
        """

        with open(filePath, encoding='UTF-8') as f:
            # 行ごとの値を要素とした配列を作成し、stripメソッドで改行記号を除く
            data = [s.strip() for s in f.readlines()]

            if lineDelimiter != '':
                # lineDelimiterでさらに要素を区切った配列を作成
                data = [l.split(lineDelimiter) for l in data]
        
        return data

    def ExportAsCSV(self, data, dirPath, fileName, header: list = []) -> None:
        """
        Export the data to CSV file.

        Parameters
        ----------
        data: array like, dict
            Data to be exported.
        dirPath: str
            Directory path (not include the file name).
        fileName: str
            File name. (not include ".csv")
        header: (Optional) list
            Header of CSV file. If list is empty, CSV file not include header.
        """
        # ----- Check directory ----- #
        self.mkdir(dirPath)

        if type(data) is dict:
            print('Exporting data...')
            for i in tqdm.tqdm(data, ncols=150):
                exportPath = dirPath + '/' + fileName + '_' + str(i) + '.csv'
                self.Write(data[i], exportPath, header)
            
        else:
            exportPath = dirPath + '/' + fileName + '.csv'
            self.Write(data, exportPath, header)
    
    def Write(self, data, path, header: list = []) -> None:
        """
        File writer

        Parameters
        ----------
        data: array like
            Data to be exported.
        path: str
            File path.
        header: (Optional) list
            Header of CSV file. If list is empty, CSV file not include header.
        """
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)

            if header:
                writer.writerow(header)
            writer.writerows(data)

    def mkdir(self, path) -> None:
        """
        Check existence of the directory, and if it does not exist, create a new one.

        Parameters
        ----------
        path: str
            Directory path
        """
        
        if not os.path.isdir(path):
            os.makedirs(path)