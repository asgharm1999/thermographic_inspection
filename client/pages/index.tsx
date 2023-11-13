import React, { useEffect, useState } from "react";

function home() {
  const [data, setData] = useState(
    "If you see this the message is not loaded yet"
  );

  useEffect(() => {
    fetch("http://localhost:8080/api/home")
      .then((response) => response.json())
      .then((data) => setData(data.message));
  }, []);

  return <div>{data}</div>;
}

export default home;
