axios.get('https://api.example.com/protected-resource', {
    headers: {
      'Authorization': '9eafd5e32ce6bdebbc2ff649cea24b1c'  // Replace with your actual API key
    }
  })
  .then(response => {
    console.log(response.data);
  })
  .catch(error => {
    if (error.response) {
      if (error.response.status === 403) {
        console.log('Access forbidden: You do not have the necessary permissions.');
      }
    } else {
      console.log('Error:', error.message);
    }
  });
  