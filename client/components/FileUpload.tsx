import React, { useState } from "react";
import styles from "./FileUpload.module.css";

const FileUpload = () => {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<
    "idle" | "uploading" | "success" | "fail"
  >("idle");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (file) {
      setStatus("uploading");

      const formData = new FormData();
      formData.append("file", file);

      try {
        const result = await fetch("http://localhost:8080/upload", {
          method: "POST",
          body: formData,
        });

        const data = await result.json();
        console.log("Upload result: ", data);

        setStatus("success");
      } catch (error) {
        console.error("Upload failed: ", error);
        setStatus("fail");
      }
    }
  };

  return (
    <>
      <div>
        <label htmlFor="file" className="sr-only">
          Choose a file
        </label>
        <input id="file" type="file" onChange={handleFileChange} className={styles.input} />
      </div>
      {file && (
        <section>
          <h2>File details:</h2>
          <ul>
            <li>Name: {file.name}</li>
            <li>Type: {file.type}</li>
            <li>Size: {file.size} bytes</li>
          </ul>
        </section>
      )}
      {file && (
        <button onClick={handleUpload} className={styles.button}>
          Upload
        </button>
      )}

      <ResultIndicator status={status} />
    </>
  );
};

const ResultIndicator = ({ status }: { status: string }) => {
  if (status === "idle") return null;
  if (status === "uploading") return <p>Uploading selected file...</p>;
  if (status === "success") return <p>Upload successful!</p>;
  if (status === "fail") return <p>Upload failed!</p>;
};

export default FileUpload;
