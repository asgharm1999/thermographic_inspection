import React from "react";
import styles from "./ProcessButton.module.css"

const ProcessButton = ({
  setResultPath,
}: {
  setResultPath: React.Dispatch<React.SetStateAction<string>>;
}) => {
  const fetchData = async () => {
    try {
      const response = await fetch("http://localhost:8080/preprocess");
      const data = await response.json();
      console.log("Process result: ", data);
      setResultPath(data.resultPath);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return <button onClick={fetchData} className={styles.button} >Process Data</button>;
};

export default ProcessButton;
