import { useState } from 'react';
import axios from 'axios';
import DragDropFile from './DragDropFile';
import Emoji from '../../layout/Emoji';
axios.defaults.baseURL = process.env.REACT_APP_FLASK_URL;

const Home = () => {
	const [audioFile, setAudioFile] = useState(null);
	const [loading, setLoading] = useState(false);
	const [emotion, setEmotion] = useState('');
	const [clicked, setClicked] = useState(false);

	const predictFile = (e) => {
		setLoading(true);
		setClicked(true);
		e.preventDefault();
		const data = new FormData();
		data.append('audioFile', audioFile);
		axios
			.post('/api', data)
			.then((res) => {
				setEmotion(res.data);
			})
			.catch((err) => {
				console.log(err);
			})
			.finally(() => {
				setLoading(false);
			});
	};

	const emojis = {
		calm: '🙂',
		disgust: '🤢',
		fearful: '😨',
		happy: '😁',
	};

	return (
		<div className=''>
			<DragDropFile
				predictFile={predictFile}
				audioFile={audioFile}
				setAudioFile={setAudioFile}
			/>
			{loading && clicked && (
				<h2 className='text-center font-bold text-[1.25em]'>
					Loading...
				</h2>
			)}
			{!loading && clicked && (
				<h2 className='font-bold text-[1.25em] flex justify-center items-center'>
					Prediction:{' '}
					<p className='mr-[1px] ml-[5px] capitalize'> {emotion}</p>{' '}
					<Emoji
						label={emotion}
						symbol={emojis[emotion.toLowerCase()]}
					/>
				</h2>
			)}
		</div>
	);
};

export default Home;
