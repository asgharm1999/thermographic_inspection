import React from "react";
import styles from "./ProcessButton.module.css";

const ProcessButton = ({
  setResultPath,
}: {
  setResultPath: React.Dispatch<React.SetStateAction<string>>;
}) => {
  const [method, setMethod] = React.useState<string>("PCT");

  const fetchData = async () => {
    try {
      const formData = new FormData();
      formData.append("method", method);

      const response = await fetch("http://localhost:8080/preprocess", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      console.log("Process result: ", data);
      setResultPath(data.resultPath);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div>
      <select onChange={(e) => setMethod(e.target.value)} className={styles.select}>
        <option value="PCT">PCT</option>
        <option value="SPCT">SPCT</option>
      </select>

      <button onClick={fetchData} className={styles.button}>
        Process Data
      </button>
    </div>
  );
};

export default ProcessButton;
